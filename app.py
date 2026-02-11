import io
import re
import zipfile
from urllib.parse import urljoin

import numpy as np
import requests
import streamlit as st
from bs4 import BeautifulSoup
from PIL import Image

# =========================
# Output size (fixed)
# =========================
TARGET_W, TARGET_H = 450, 633
TARGET_AR = TARGET_W / TARGET_H

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
    )
}


# =========================
# Utils
# =========================
def safe_name(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^\w\-.가-힣]+", "_", s)
    return s[:120] if s else "item"


def download_image(url: str) -> Image.Image:
    r = requests.get(url, headers={**HEADERS, "Referer": url}, timeout=30)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")


# =========================
# 1) Split long detail image into pieces by horizontal white gaps
# =========================
def split_detail_image_by_white_rows(
    pil_img: Image.Image,
    white_thr: int = 245,
    white_ratio: float = 0.985,
    min_gap: int = 70,
    min_cut_height: int = 220,
):
    """
    긴 상세페이지 이미지(세로로 매우 긴 경우)에서
    가로 '화이트 갭'을 찾아 적당히 분할합니다.
    """
    gray = pil_img.convert("L")
    arr = np.array(gray)
    h, w = arr.shape
    row_white_ratio = (arr > white_thr).mean(axis=1)

    gaps = []
    in_gap = False
    start = 0

    for i, r in enumerate(row_white_ratio):
        if r >= white_ratio:
            if not in_gap:
                in_gap = True
                start = i
        else:
            if in_gap and i - start >= min_gap:
                gaps.append((start, i))
            in_gap = False

    if in_gap and h - start >= min_gap:
        gaps.append((start, h))

    cuts = []
    prev = 0
    for g1, g2 in gaps:
        if g1 - prev >= min_cut_height:
            cuts.append((prev, g1))
        prev = g2

    if h - prev >= min_cut_height:
        cuts.append((prev, h))

    return cuts


def should_split(pil_img: Image.Image) -> bool:
    """
    '상세페이지 긴 이미지'만 분할 적용.
    """
    w, h = pil_img.size
    return h >= int(w * 2.0)


# =========================
# 2) Edge whitespace trim (true cropping)
# =========================
def trim_edge_whitespace(
    pil_img: Image.Image,
    white_thr: int = 245,
    white_ratio: float = 0.99,
    min_run: int = 10,
):
    """
    이미지 가장자리(상하좌우)에서 '거의 흰색'인 구간을 실제로 잘라냅니다.
    - white_ratio를 조금 올려(0.99) 얇은 보더도 더 강하게 제거
    - min_run을 늘려(10) 아주 얇은 라인도 잘 제거되도록
    """
    img = pil_img.convert("RGB")
    arr = np.array(img)
    h, w = arr.shape[:2]

    def row_is_white(y: int) -> bool:
        row = arr[y, :, :]
        return (row > white_thr).all(axis=1).mean() >= white_ratio

    def col_is_white(x: int) -> bool:
        col = arr[:, x, :]
        return (col > white_thr).all(axis=1).mean() >= white_ratio

    top, bottom = 0, h - 1
    left, right = 0, w - 1

    # 반복 트림
    while True:
        changed = False

        run = 0
        while top < bottom and row_is_white(top):
            top += 1
            run += 1
        if run >= min_run:
            changed = True

        run = 0
        while bottom > top and row_is_white(bottom):
            bottom -= 1
            run += 1
        if run >= min_run:
            changed = True

        run = 0
        while left < right and col_is_white(left):
            left += 1
            run += 1
        if run >= min_run:
            changed = True

        run = 0
        while right > left and col_is_white(right):
            right -= 1
            run += 1
        if run >= min_run:
            changed = True

        # 과도한 트림 방지
        if (bottom - top) < 120 or (right - left) < 120:
            return img

        if not changed:
            break

        arr = arr[top : bottom + 1, left : right + 1, :]
        img = Image.fromarray(arr)
        arr = np.array(img)
        h, w = arr.shape[:2]
        top, bottom, left, right = 0, h - 1, 0, w - 1

    return img


# =========================
# 3) Foreground detection (bbox + center)
# =========================
def estimate_background_color(arr_rgb: np.ndarray) -> np.ndarray:
    """
    모서리 샘플의 중앙값으로 배경색을 추정합니다.
    """
    h, w = arr_rgb.shape[:2]
    corners = np.array(
        [arr_rgb[0, 0], arr_rgb[0, w - 1], arr_rgb[h - 1, 0], arr_rgb[h - 1, w - 1]],
        dtype=np.int16,
    )
    return np.median(corners, axis=0)


def foreground_mask(arr_rgb: np.ndarray, diff_thr: int = 24) -> np.ndarray:
    """
    배경 추정 색상과의 색 차이를 이용해 전경 마스크를 만듭니다.
    diff_thr를 살짝 상향(24)해 '흰 배경'에서 얇은 노이즈를 덜 전경으로 잡게 합니다.
    """
    arr = arr_rgb.astype(np.int16)
    bg = estimate_background_color(arr)
    diff = np.sqrt(((arr - bg) ** 2).sum(axis=2))
    return diff > diff_thr


def foreground_bbox_center(pil_img: Image.Image, diff_thr: int = 24):
    """
    전경 마스크의 bbox(경계상자)와 중심을 반환합니다.
    - bbox 기반으로 크롭하면 피사체가 더 중앙에 안정적으로 위치합니다.
    """
    img = pil_img.convert("RGB")
    arr = np.array(img)
    h, w = arr.shape[:2]

    mask = foreground_mask(arr, diff_thr=diff_thr)

    if not mask.any():
        # 전경을 못 잡으면 중앙
        return (0, 0, w - 1, h - 1), (w / 2.0, h / 2.0)

    ys, xs = np.where(mask)
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return (x1, y1, x2, y2), (cx, cy)


# =========================
# 4) Crop strategy (bbox-fit + aspect fill)
# =========================
def crop_around_bbox_to_aspect(
    pil_img: Image.Image,
    target_ar: float,
    bbox,
    center,
    pad_ratio: float = 0.18,
):
    """
    핵심:
    - 전경 bbox를 기준으로 '적당한 여유(pad_ratio)'를 주고
    - 그 영역을 target_ar에 맞춰 확장(필요 시)해서 자릅니다.
    - 절대 패딩(여백 추가) 없이 "자르기"만 합니다.
    => 흰 여백이 보일 가능성이 크게 줄어듭니다.
    """
    img = pil_img.convert("RGB")
    W, H = img.size
    x1, y1, x2, y2 = bbox
    cx, cy = center

    bw = max(1, (x2 - x1))
    bh = max(1, (y2 - y1))

    # bbox에 여유를 조금 주기(피사체가 너무 꽉 차서 잘리는 느낌 방지)
    pad_w = int(round(bw * pad_ratio))
    pad_h = int(round(bh * pad_ratio))

    rx1 = int(max(0, x1 - pad_w))
    ry1 = int(max(0, y1 - pad_h))
    rx2 = int(min(W - 1, x2 + pad_w))
    ry2 = int(min(H - 1, y2 + pad_h))

    rw = max(1, rx2 - rx1)
    rh = max(1, ry2 - ry1)

    # 이제 이 '관심영역'을 target_ar에 맞춰 "확장" (줄이지 않음)
    # 확장할 때도 중심은 전경 중심(cx,cy)을 최대한 유지
    roi_ar = rw / rh
    if roi_ar > target_ar:
        # 너무 가로로 넓음 -> 높이를 늘려야 함
        crop_w = rw
        crop_h = int(round(crop_w / target_ar))
    else:
        # 너무 세로로 김 -> 너비를 늘려야 함
        crop_h = rh
        crop_w = int(round(target_ar * crop_h))

    # 전경 중심 기준으로 배치
    left = int(round(cx - crop_w / 2))
    top = int(round(cy - crop_h / 2))

    # 이미지 경계 안으로 클램프
    left = max(0, min(left, W - crop_w))
    top = max(0, min(top, H - crop_h))

    return img.crop((left, top, left + crop_w, top + crop_h))


def final_safe_trim_after_resize(pil_img: Image.Image):
    """
    최종 450x633로 리사이즈 후, 혹시 남아있을 수 있는 1~3px 수준의
    얇은 흰 테두리를 '미세 트림'으로 제거한 뒤 다시 450x633으로 맞춥니다.
    (사용자 요구: 흰 여백 절대 보이면 안됨)
    """
    # 아주 얇은 테두리만 제거하려고 min_run 작게
    trimmed = trim_edge_whitespace(
        pil_img,
        white_thr=247,
        white_ratio=0.995,
        min_run=2,
    )
    if trimmed.size != (TARGET_W, TARGET_H):
        trimmed = trimmed.resize((TARGET_W, TARGET_H), Image.LANCZOS)
    return trimmed


def make_thumb_450x633(pil_img: Image.Image):
    # 1) 가장자리 흰 여백 제거(1차)
    cut = trim_edge_whitespace(pil_img)

    # 2) 전경 bbox/center 계산
    bbox, center = foreground_bbox_center(cut, diff_thr=24)

    # 3) bbox 기반 + target_ar 맞춤 크롭 (피사체 중앙 정렬 강화)
    cropped = crop_around_bbox_to_aspect(
        cut,
        TARGET_AR,
        bbox=bbox,
        center=center,
        pad_ratio=0.18,
    )

    # 4) 최종 리사이즈 (패딩 없음)
    out = cropped.resize((TARGET_W, TARGET_H), Image.LANCZOS)

    # 5) 혹시 남을 미세 흰 테두리까지 최종 안전 트림
    out = final_safe_trim_after_resize(out)
    return out


# =========================
# 5) URL: extract ONLY from detail content area
# =========================
DETAIL_CONTAINER_SELECTORS = [
    "#prdDetailContent",
    "#prdDetail",
    ".xans-product-detail",
    ".xans-product-detaildesign",
    ".xans-product-additional",
    "#productDetail",
]


def extract_detail_image_urls_only(page_url: str, max_images: int = 250) -> list[str]:
    """
    ✅ 상세페이지 URL 입력 시:
    '본문 상세영역 컨테이너' 안에 있는 img만 수집
    """
    html = requests.get(page_url, headers=HEADERS, timeout=25).text
    soup = BeautifulSoup(html, "lxml")

    container = None
    for sel in DETAIL_CONTAINER_SELECTORS:
        container = soup.select_one(sel)
        if container:
            break

    # 본문 컨테이너를 못 찾으면(테마 이슈) -> 전체에서 찾되, 마지막 fallback
    scope = container if container else soup

    urls = []
    for img in scope.select("img"):
        src = (
            img.get("src")
            or img.get("data-src")
            or img.get("data-original")
            or ""
        ).strip()
        if not src:
            continue

        full = urljoin(page_url, src)

        # data URI 제외
        if full.startswith("data:"):
            continue

        if full not in urls:
            urls.append(full)

        if len(urls) >= max_images:
            break

    return urls


# =========================
# Processing
# =========================
def process_image_any(pil_img: Image.Image, prefix: str):
    outputs = []

    if should_split(pil_img):
        segments = split_detail_image_by_white_rows(pil_img)
        if not segments:
            segments = [(0, pil_img.height)]
        for idx, (y1, y2) in enumerate(segments, start=1):
            piece = pil_img.crop((0, y1, pil_img.width, y2))
            thumb = make_thumb_450x633(piece)
            outputs.append((f"{prefix}_{idx:02d}_{TARGET_W}x{TARGET_H}.jpg", thumb))
    else:
        thumb = make_thumb_450x633(pil_img)
        outputs.append((f"{prefix}_01_{TARGET_W}x{TARGET_H}.jpg", thumb))

    return outputs


# =========================
# Streamlit UI
# =========================
st.set_page_config(layout="wide")

# ---- Title (30% smaller) + English subtitle ----
st.markdown(
    """
    <style>
      .misharp-title-wrap { margin-top: 8px; margin-bottom: 6px; }
      .misharp-title { font-size: 1.55rem; font-weight: 800; letter-spacing: -0.02em; margin: 0; }
      .misharp-sub { font-size: 0.78rem; color: #666; margin-top: 6px; }
      .misharp-caption { color:#666; font-size: 0.92rem; margin-top: 8px; }
    </style>
    <div class="misharp-title-wrap">
      <div class="misharp-title">MISHARP 상세페이지 썸네일 생성기</div>
      <div class="misharp-sub">MISHARP THUMBNAIL GENERATOR V1</div>
      <div class="misharp-caption">URL 입력 시 본문(상세영역) 이미지에서만 추출 → 450×633 여백 없이 피사체 중앙 크롭</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("고급 옵션", expanded=False):
    max_images = st.slider("상세영역에서 수집할 최대 이미지 수", 50, 600, 250, step=50)
    st.write("※ URL 입력: 본문 상세영역(#prdDetail 등) 내부 img만 수집합니다.")
    st.write("※ 크롭 방식: 전경(피사체) bbox 기반 중앙정렬 + 패딩 없이 크롭(흰 여백 방지).")

tab1, tab2, tab3 = st.tabs(["① 상세페이지 URL", "② 이미지 주소(URL)", "③ 이미지 업로드"])

all_outputs = []

# ① 상세페이지 URL (본문 전용)
with tab1:
    page_url = st.text_input("상세페이지 URL", placeholder="https://.../product/detail.html?product_no=28461")
    if st.button("URL에서 '본문 상세이미지'만 수집 → 생성", type="primary", key="go1"):
        if not page_url.strip():
            st.error("상세페이지 URL을 입력해주세요.")
        else:
            with st.spinner("본문 상세영역 이미지 URL 수집 중…"):
                urls = extract_detail_image_urls_only(page_url.strip(), max_images=max_images)

            if not urls:
                st.error("본문(상세영역)에서 이미지 URL을 찾지 못했습니다. 테마 구조가 다를 수 있어요.")
            else:
                with st.spinner(f"다운로드 및 처리 중… ({len(urls)}개)"):
                    for i, u in enumerate(urls, start=1):
                        try:
                            pil = download_image(u)
                            all_outputs += process_image_any(pil, f"url{i:03d}")
                        except Exception:
                            continue

# ② 이미지 주소
with tab2:
    st.write("이미지 URL을 여러 줄로 붙여넣으세요. (각 줄 1개)")
    url_text = st.text_area("이미지 주소 목록", height=180, placeholder="https://.../a.jpg\nhttps://.../b.jpg\n...")
    if st.button("이미지 주소로 생성", type="primary", key="go2"):
        lines = [l.strip() for l in (url_text or "").splitlines() if l.strip()]
        if not lines:
            st.error("이미지 URL을 넣어주세요.")
        else:
            with st.spinner(f"다운로드 및 처리 중… ({len(lines)}개)"):
                for i, u in enumerate(lines, start=1):
                    try:
                        pil = download_image(u)
                        all_outputs += process_image_any(pil, f"img{i:03d}")
                    except Exception:
                        continue

# ③ 업로드
with tab3:
    uploads = st.file_uploader(
        "상세페이지 이미지 업로드 (여러 장 가능)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
    )
    if uploads:
        with st.spinner(f"업로드 이미지 처리 중… ({len(uploads)}개)"):
            for i, f in enumerate(uploads, start=1):
                try:
                    pil = Image.open(f).convert("RGB")
                    base = safe_name(f.name.rsplit(".", 1)[0])
                    all_outputs += process_image_any(pil, f"up{i:03d}_{base}")
                except Exception:
                    continue

# 결과 출력
if all_outputs:
    fixed = []
    for name, img in all_outputs:
        # 최종 규격 강제 + 안전 트림
        img = img.resize((TARGET_W, TARGET_H), Image.LANCZOS)
        img = final_safe_trim_after_resize(img)
        fixed.append((name, img))

    st.success(f"총 {len(fixed)}장 생성 완료 (모두 {TARGET_W}×{TARGET_H}, 흰 여백 0 목표)")
    st.subheader("미리보기 (일부)")
    st.image([img for _, img in fixed[:24]], width=180)

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, img in fixed:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=95)
            zf.writestr(name, buf.getvalue())
    zip_buf.seek(0)

    st.download_button(
        "ZIP 다운로드 (450×633)",
        data=zip_buf,
        file_name=f"thumb_{TARGET_W}x{TARGET_H}.zip",
        mime="application/zip",
    )
else:
    st.info("아직 결과가 없습니다. 위 탭에서 입력 후 생성해보세요.")

st.markdown(
    """
    <hr style="margin-top:40px; margin-bottom:10px;">
    <div style="font-size:11px; color:#888; line-height:1.5; text-align:center;">
        ⓒ misharpcompany. All rights reserved.<br>
        본 프로그램의 저작권은 미샵컴퍼니(misharpcompany)에 있으며, 무단 복제·배포·사용을 금합니다.<br>
        본 프로그램은 미샵컴퍼니 내부 직원 전용으로, 외부 유출 및 제3자 제공을 엄격히 금합니다.
        <br><br>
        ⓒ misharpcompany. All rights reserved.<br>
        This program is the intellectual property of misharpcompany.
        Unauthorized copying, distribution, or use is strictly prohibited.<br>
        This program is for internal use by misharpcompany employees only
        and must not be disclosed or shared externally.
    </div>
    """,
    unsafe_allow_html=True
)
