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
# Solid band detection (WHITE / BLACK)
# =========================
def _row_solid_ratio_rgb(arr_rgb: np.ndarray, y: int, mode: str):
    """
    mode: 'white' or 'black'
    """
    row = arr_rgb[y, :, :].astype(np.int16)
    # 얼마나 "거의 같은 색"인가 (노이즈 많은 사진 영역 제외)
    row_std = row.std(axis=0).mean()
    row_mean = row.mean()

    if mode == "white":
        # 대부분이 아주 밝고, 균일하면 흰 띠로 판정
        solid = (row > 245).all(axis=1).mean()
        return solid, row_mean, row_std
    else:
        # 대부분이 아주 어둡고, 균일하면 검정 띠로 판정
        solid = (row < 12).all(axis=1).mean()
        return solid, row_mean, row_std


def _col_solid_ratio_rgb(arr_rgb: np.ndarray, x: int, mode: str):
    col = arr_rgb[:, x, :].astype(np.int16)
    col_std = col.std(axis=0).mean()
    col_mean = col.mean()

    if mode == "white":
        solid = (col > 245).all(axis=1).mean()
        return solid, col_mean, col_std
    else:
        solid = (col < 12).all(axis=1).mean()
        return solid, col_mean, col_std


def trim_edge_bands(
    pil_img: Image.Image,
    solid_ratio_thr: float = 0.985,
    std_thr: float = 8.0,
    min_run: int = 8,
):
    """
    ✅ 가장자리의 '흰 띠' / '검정 띠'를 실제로 잘라냅니다.
    - solid_ratio_thr: 행/열 픽셀의 몇 %가 거의 흰/검정이면 띠로 볼지
    - std_thr: 색 변화가 거의 없는(균일한) 띠만 제거(사진 영역 오검출 방지)
    - min_run: 최소 몇 줄 이상 연속될 때만 잘라냄
    """
    img = pil_img.convert("RGB")
    arr = np.array(img)
    h, w = arr.shape[:2]

    def row_is_band(y: int) -> bool:
        # white OR black band
        solid_w, mean_w, std_w = _row_solid_ratio_rgb(arr, y, "white")
        solid_b, mean_b, std_b = _row_solid_ratio_rgb(arr, y, "black")
        return (solid_w >= solid_ratio_thr and std_w <= std_thr) or (
            solid_b >= solid_ratio_thr and std_b <= std_thr
        )

    def col_is_band(x: int) -> bool:
        solid_w, mean_w, std_w = _col_solid_ratio_rgb(arr, x, "white")
        solid_b, mean_b, std_b = _col_solid_ratio_rgb(arr, x, "black")
        return (solid_w >= solid_ratio_thr and std_w <= std_thr) or (
            solid_b >= solid_ratio_thr and std_b <= std_thr
        )

    top, bottom = 0, h - 1
    left, right = 0, w - 1

    while True:
        changed = False

        run = 0
        while top < bottom and row_is_band(top):
            top += 1
            run += 1
        if run >= min_run:
            changed = True

        run = 0
        while bottom > top and row_is_band(bottom):
            bottom -= 1
            run += 1
        if run >= min_run:
            changed = True

        run = 0
        while left < right and col_is_band(left):
            left += 1
            run += 1
        if run >= min_run:
            changed = True

        run = 0
        while right > left and col_is_band(right):
            right -= 1
            run += 1
        if run >= min_run:
            changed = True

        # 과도한 트림 방지
        if (bottom - top) < 140 or (right - left) < 140:
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
# Split long detail image into pieces by WHITE/BLACK gaps
# =========================
def split_detail_image_by_solid_rows(
    pil_img: Image.Image,
    white_thr: int = 245,
    black_thr: int = 12,
    solid_ratio: float = 0.985,
    min_gap: int = 70,
    min_cut_height: int = 220,
    std_thr: float = 10.0,
):
    """
    ✅ 긴 상세이미지를 '흰 갭' 뿐 아니라 '검정 갭(검정 구분띠)'도 컷 포인트로 사용.
    """
    img = pil_img.convert("RGB")
    arr = np.array(img).astype(np.int16)
    h, w = arr.shape[:2]

    # row마다 "거의 흰색" 비율 / "거의 검정" 비율 / 표준편차(균일성)
    row_white = (arr > white_thr).all(axis=2).mean(axis=1)
    row_black = (arr < black_thr).all(axis=2).mean(axis=1)
    row_std = arr.std(axis=1).mean(axis=1)

    is_gap = ((row_white >= solid_ratio) | (row_black >= solid_ratio)) & (row_std <= std_thr)

    gaps = []
    in_gap = False
    start = 0
    for i, g in enumerate(is_gap):
        if g and not in_gap:
            in_gap = True
            start = i
        elif (not g) and in_gap:
            if i - start >= min_gap:
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
    w, h = pil_img.size
    return h >= int(w * 2.0)


# =========================
# Foreground detection (bbox + center) with "band ignore"
# =========================
def estimate_background_color(arr_rgb: np.ndarray) -> np.ndarray:
    h, w = arr_rgb.shape[:2]
    corners = np.array(
        [arr_rgb[0, 0], arr_rgb[0, w - 1], arr_rgb[h - 1, 0], arr_rgb[h - 1, w - 1]],
        dtype=np.int16,
    )
    return np.median(corners, axis=0)


def foreground_mask(arr_rgb: np.ndarray, diff_thr: int = 24) -> np.ndarray:
    arr = arr_rgb.astype(np.int16)
    bg = estimate_background_color(arr)
    diff = np.sqrt(((arr - bg) ** 2).sum(axis=2))
    return diff > diff_thr


def foreground_bbox_center(pil_img: Image.Image, diff_thr: int = 24):
    """
    ✅ 전경 bbox를 잡을 때, 가장자리 band(흰/검정 띠)가 남아있다면
    먼저 trim_edge_bands()로 제거하고 들어가도록 위에서 처리합니다.
    """
    img = pil_img.convert("RGB")
    arr = np.array(img)
    h, w = arr.shape[:2]

    mask = foreground_mask(arr, diff_thr=diff_thr)
    if not mask.any():
        return (0, 0, w - 1, h - 1), (w / 2.0, h / 2.0)

    ys, xs = np.where(mask)
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return (x1, y1, x2, y2), (cx, cy)


# =========================
# Crop strategy (bbox-fit + aspect fill)
# =========================
def crop_around_bbox_to_aspect(
    pil_img: Image.Image,
    target_ar: float,
    bbox,
    center,
    pad_ratio: float = 0.16,
):
    img = pil_img.convert("RGB")
    W, H = img.size
    x1, y1, x2, y2 = bbox
    cx, cy = center

    bw = max(1, (x2 - x1))
    bh = max(1, (y2 - y1))

    pad_w = int(round(bw * pad_ratio))
    pad_h = int(round(bh * pad_ratio))

    rx1 = int(max(0, x1 - pad_w))
    ry1 = int(max(0, y1 - pad_h))
    rx2 = int(min(W - 1, x2 + pad_w))
    ry2 = int(min(H - 1, y2 + pad_h))

    rw = max(1, rx2 - rx1)
    rh = max(1, ry2 - ry1)

    roi_ar = rw / rh
    if roi_ar > target_ar:
        crop_w = rw
        crop_h = int(round(crop_w / target_ar))
    else:
        crop_h = rh
        crop_w = int(round(target_ar * crop_h))

    # 중심 기준 배치
    left = int(round(cx - crop_w / 2))
    top = int(round(cy - crop_h / 2))

    # 클램프
    crop_w = min(crop_w, W)
    crop_h = min(crop_h, H)

    left = max(0, min(left, W - crop_w))
    top = max(0, min(top, H - crop_h))

    return img.crop((left, top, left + crop_w, top + crop_h))


def final_safe_trim_after_resize(pil_img: Image.Image):
    """
    ✅ 최종 450×633 리사이즈 후:
    혹시 남은 1~3px 수준의 흰/검정 테두리까지 제거 → 다시 450×633
    """
    trimmed = trim_edge_bands(
        pil_img,
        solid_ratio_thr=0.992,
        std_thr=9.0,
        min_run=2,
    )
    if trimmed.size != (TARGET_W, TARGET_H):
        trimmed = trimmed.resize((TARGET_W, TARGET_H), Image.LANCZOS)
    return trimmed


def make_thumb_450x633(pil_img: Image.Image):
    # (A) 먼저 흰/검정 띠 제거 (가장 중요)
    cut = trim_edge_bands(pil_img)

    # (B) 전경 bbox/center
    bbox, center = foreground_bbox_center(cut, diff_thr=24)

    # (C) bbox 기반 중앙 크롭
    cropped = crop_around_bbox_to_aspect(
        cut,
        TARGET_AR,
        bbox=bbox,
        center=center,
        pad_ratio=0.16,
    )

    # (D) 최종 리사이즈
    out = cropped.resize((TARGET_W, TARGET_H), Image.LANCZOS)

    # (E) 마지막 안전 트림(흰/검정 모두)
    out = final_safe_trim_after_resize(out)
    return out


# =========================
# URL: extract ONLY from detail content area
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
    html = requests.get(page_url, headers=HEADERS, timeout=25).text
    soup = BeautifulSoup(html, "lxml")

    container = None
    for sel in DETAIL_CONTAINER_SELECTORS:
        container = soup.select_one(sel)
        if container:
            break

    scope = container if container else soup

    urls = []
    for img in scope.select("img"):
        src = (img.get("src") or img.get("data-src") or img.get("data-original") or "").strip()
        if not src:
            continue

        full = urljoin(page_url, src)
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
        segments = split_detail_image_by_solid_rows(pil_img)
        if not segments:
            segments = [(0, pil_img.height)]

        for idx, (y1, y2) in enumerate(segments, start=1):
            piece = pil_img.crop((0, y1, pil_img.width, y2))
            # 조각도 먼저 띠 제거
            piece = trim_edge_bands(piece)
            thumb = make_thumb_450x633(piece)
            outputs.append((f"{prefix}_{idx:02d}_{TARGET_W}x{TARGET_H}.jpg", thumb))
    else:
        pil_img = trim_edge_bands(pil_img)
        thumb = make_thumb_450x633(pil_img)
        outputs.append((f"{prefix}_01_{TARGET_W}x{TARGET_H}.jpg", thumb))

    return outputs


# =========================
# Streamlit UI
# =========================
st.set_page_config(layout="wide")

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
      <div class="misharp-caption">본문 상세영역 이미지에서만 추출 → 450×633 / 흰·검정 띠 제거 / 피사체 중앙 크롭</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("고급 옵션", expanded=False):
    max_images = st.slider("상세영역에서 수집할 최대 이미지 수", 50, 600, 250, step=50)
    st.write("※ 검정 바(하단 검정 띠) 문제를 막기 위해: 가장자리 '흰/검정 띠'를 자동 제거합니다.")
    st.write("※ 긴 이미지 분할 시: 흰 갭 + 검정 갭 모두 컷 포인트로 사용합니다.")

tab1, tab2, tab3 = st.tabs(["① 상세페이지 URL", "② 이미지 주소(URL)", "③ 이미지 업로드"])

all_outputs = []

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

if all_outputs:
    fixed = []
    for name, img in all_outputs:
        img = img.resize((TARGET_W, TARGET_H), Image.LANCZOS)
        img = final_safe_trim_after_resize(img)
        fixed.append((name, img))

    st.success(f"총 {len(fixed)}장 생성 완료 (모두 {TARGET_W}×{TARGET_H}, 검정/흰 띠 자동 제거)")
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
