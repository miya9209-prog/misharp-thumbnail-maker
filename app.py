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
# Edge band trim (white/black) - conservative
# =========================
def trim_edge_bands(
    pil_img: Image.Image,
    solid_ratio_thr: float = 0.992,
    std_thr: float = 7.5,
    min_run: int = 10,
    white_thr: int = 246,
    black_thr: int = 9,
):
    img = pil_img.convert("RGB")
    arr = np.array(img).astype(np.int16)
    h, w = arr.shape[:2]

    def row_is_band(y: int) -> bool:
        row = arr[y, :, :]
        row_std = row.std(axis=0).mean()
        white_ratio = (row > white_thr).all(axis=1).mean()
        black_ratio = (row < black_thr).all(axis=1).mean()
        return row_std <= std_thr and (white_ratio >= solid_ratio_thr or black_ratio >= solid_ratio_thr)

    def col_is_band(x: int) -> bool:
        col = arr[:, x, :]
        col_std = col.std(axis=0).mean()
        white_ratio = (col > white_thr).all(axis=1).mean()
        black_ratio = (col < black_thr).all(axis=1).mean()
        return col_std <= std_thr and (white_ratio >= solid_ratio_thr or black_ratio >= solid_ratio_thr)

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

        # 과도 트림 방지
        if (bottom - top) < 200 or (right - left) < 200:
            return img

        if not changed:
            break

        arr = arr[top : bottom + 1, left : right + 1, :]
        img = Image.fromarray(arr.astype(np.uint8))
        arr = np.array(img).astype(np.int16)
        h, w = arr.shape[:2]
        top, bottom, left, right = 0, h - 1, 0, w - 1

    return img


# =========================
# NEW: Micro edge strip (1~3px) for tiny white lines
# =========================
def micro_strip_lines(
    pil_img: Image.Image,
    max_strip: int = 3,
    white_thr: int = 247,
    black_thr: int = 8,
    ratio_thr: float = 0.97,
):
    """
    ✅ 최종 450×633에서 생기는 1~3px 미세 흰줄/검정줄 제거용.
    - max_strip 이내에서만 제거 (과도 크롭 방지)
    - ratio_thr: 해당 라인이 거의 전부 흰색/검정이면 제거
    """
    img = pil_img.convert("RGB")
    arr = np.array(img).astype(np.int16)
    h, w = arr.shape[:2]

    def row_is_micro(y: int) -> bool:
        row = arr[y, :, :]
        white_ratio = (row > white_thr).all(axis=1).mean()
        black_ratio = (row < black_thr).all(axis=1).mean()
        return white_ratio >= ratio_thr or black_ratio >= ratio_thr

    def col_is_micro(x: int) -> bool:
        col = arr[:, x, :]
        white_ratio = (col > white_thr).all(axis=1).mean()
        black_ratio = (col < black_thr).all(axis=1).mean()
        return white_ratio >= ratio_thr or black_ratio >= ratio_thr

    top, bottom = 0, h - 1
    left, right = 0, w - 1

    # 위
    stripped = 0
    while stripped < max_strip and top < bottom and row_is_micro(top):
        top += 1
        stripped += 1

    # 아래
    stripped = 0
    while stripped < max_strip and bottom > top and row_is_micro(bottom):
        bottom -= 1
        stripped += 1

    # 좌
    stripped = 0
    while stripped < max_strip and left < right and col_is_micro(left):
        left += 1
        stripped += 1

    # 우
    stripped = 0
    while stripped < max_strip and right > left and col_is_micro(right):
        right -= 1
        stripped += 1

    # 너무 작아지면 원본 반환
    if (bottom - top) < 200 or (right - left) < 200:
        return img

    if top == 0 and bottom == h - 1 and left == 0 and right == w - 1:
        return img

    cropped = Image.fromarray(arr[top : bottom + 1, left : right + 1, :].astype(np.uint8))
    return cropped


# =========================
# Split long detail image by solid rows (white/black gaps)
# =========================
def should_split(pil_img: Image.Image) -> bool:
    w, h = pil_img.size
    return h >= int(w * 2.0)


def split_detail_image_by_solid_rows(
    pil_img: Image.Image,
    white_thr: int = 246,
    black_thr: int = 9,
    solid_ratio: float = 0.992,
    min_gap: int = 70,
    min_cut_height: int = 220,
    std_thr: float = 9.0,
):
    img = pil_img.convert("RGB")
    arr = np.array(img).astype(np.int16)
    h, w = arr.shape[:2]

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


# =========================
# Subject center estimation (center shift only)
# =========================
def estimate_background_color(arr_rgb: np.ndarray) -> np.ndarray:
    h, w = arr_rgb.shape[:2]
    corners = np.array(
        [arr_rgb[0, 0], arr_rgb[0, w - 1], arr_rgb[h - 1, 0], arr_rgb[h - 1, w - 1]],
        dtype=np.int16,
    )
    return np.median(corners, axis=0)


def subject_center(pil_img: Image.Image, diff_thr: int = 26):
    img = pil_img.convert("RGB")
    arr = np.array(img).astype(np.int16)
    h, w = arr.shape[:2]
    bg = estimate_background_color(arr)
    diff = np.sqrt(((arr - bg) ** 2).sum(axis=2))
    mask = diff > diff_thr
    if not mask.any():
        return (w / 2.0, h / 2.0)
    ys, xs = np.where(mask)
    return (float(xs.mean()), float(ys.mean()))


# =========================
# NO DISTORTION: Uniform cover resize then crop
# =========================
def resize_cover_then_crop(pil_img: Image.Image, center_xy=None):
    img = pil_img.convert("RGB")
    W, H = img.size

    scale = max(TARGET_W / W, TARGET_H / H)
    new_w = int(round(W * scale))
    new_h = int(round(H * scale))

    resized = img.resize((new_w, new_h), Image.LANCZOS)

    if center_xy is None:
        cx, cy = (new_w / 2.0, new_h / 2.0)
    else:
        ox, oy = center_xy
        cx, cy = (ox * scale, oy * scale)

    left = int(round(cx - TARGET_W / 2.0))
    top = int(round(cy - TARGET_H / 2.0))

    left = max(0, min(left, new_w - TARGET_W))
    top = max(0, min(top, new_h - TARGET_H))

    return resized.crop((left, top, left + TARGET_W, top + TARGET_H))


def make_thumb_450x633(pil_img: Image.Image):
    # (A) 보수적 띠 제거(큰 흰/검정 바)
    cut = trim_edge_bands(pil_img)

    # (B) 중심점 계산(확대/축소에 관여 X)
    cxy = subject_center(cut, diff_thr=26)

    # (C) 비율 유지 cover + 중심 이동 크롭
    out = resize_cover_then_crop(cut, center_xy=cxy)

    # (D) ✅ 미세 흰줄/검정줄(1~3px) 제거
    out2 = micro_strip_lines(out, max_strip=3, ratio_thr=0.97)

    # (E) 크기가 줄었으면 “같은 원칙(cover)”으로 다시 450×633 복구 (비율 왜곡 금지)
    if out2.size != (TARGET_W, TARGET_H):
        out2 = resize_cover_then_crop(out2, center_xy=subject_center(out2, diff_thr=26))

    # (F) 마지막으로 한 번 더 micro strip (보간으로 새로 생기는 1px 라인 방지)
    out3 = micro_strip_lines(out2, max_strip=2, ratio_thr=0.975)
    if out3.size != (TARGET_W, TARGET_H):
        out3 = resize_cover_then_crop(out3, center_xy=subject_center(out3, diff_thr=26))

    return out3


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
      <div class="misharp-caption">비율 왜곡(늘림/비틀림) 없이: Cover(꽉채움) + 중심 이동 크롭 + 미세 흰줄(1~3px) 제거</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("고급 옵션", expanded=False):
    max_images = st.slider("상세영역에서 수집할 최대 이미지 수", 50, 600, 250, step=50)
    st.write("※ 변형 금지: 비율 유지(Uniform) 리사이즈만 사용합니다.")
    st.write("※ 여백 금지: 450×633은 Cover(꽉채움) 방식으로만 생성합니다.")
    st.write("※ 미세 흰줄 제거: 최종 결과에서 1~3px 라인을 추가로 제거합니다.")

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
    st.success(f"총 {len(all_outputs)}장 생성 완료 (모두 {TARGET_W}×{TARGET_H}, 미세 흰줄 제거 포함)")
    st.subheader("미리보기 (일부)")
    st.image([img for _, img in all_outputs[:24]], width=180)

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, img in all_outputs:
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
