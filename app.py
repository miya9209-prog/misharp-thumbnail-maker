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
    white_ratio: float = 0.985,
    min_run: int = 6,
):
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

    changed = True
    while changed:
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
        if (bottom - top) < 80 or (right - left) < 80:
            return img

        arr = arr[top : bottom + 1, left : right + 1, :]
        h, w = arr.shape[:2]
        top, bottom, left, right = 0, h - 1, 0, w - 1

    return Image.fromarray(arr)


# =========================
# 3) Foreground center (subject center)
# =========================
def foreground_center(pil_img: Image.Image, diff_thr: int = 22):
    img = pil_img.convert("RGB")
    arr = np.array(img).astype(np.int16)
    h, w = arr.shape[:2]

    corners = np.array(
        [arr[0, 0], arr[0, w - 1], arr[h - 1, 0], arr[h - 1, w - 1]],
        dtype=np.int16,
    )
    bg = np.median(corners, axis=0)

    diff = np.sqrt(((arr - bg) ** 2).sum(axis=2))
    mask = diff > diff_thr

    if not mask.any():
        return (w / 2.0, h / 2.0)

    ys, xs = np.where(mask)
    return (xs.mean(), ys.mean())


# =========================
# 4) Fill-crop to target aspect (no padding, no blur)
# =========================
def fill_crop_to_aspect(pil_img: Image.Image, target_ar: float, cx: float, cy: float):
    img = pil_img.convert("RGB")
    W, H = img.size
    ar = W / H

    if ar > target_ar:
        crop_h = H
        crop_w = int(round(target_ar * crop_h))
    else:
        crop_w = W
        crop_h = int(round(crop_w / target_ar))

    left = int(round(cx - crop_w / 2))
    top = int(round(cy - crop_h / 2))

    left = max(0, min(left, W - crop_w))
    top = max(0, min(top, H - crop_h))

    return img.crop((left, top, left + crop_w, top + crop_h))


def make_thumb_450x633(pil_img: Image.Image):
    cut = trim_edge_whitespace(pil_img)
    cx, cy = foreground_center(cut, diff_thr=22)
    cropped = fill_crop_to_aspect(cut, TARGET_AR, cx, cy)
    return cropped.resize((TARGET_W, TARGET_H), Image.LANCZOS)


# =========================
# 5) URL: extract ONLY from detail content area
# =========================
DETAIL_CONTAINER_SELECTORS = [
    "#prdDetailContent",
    "#prdDetail",
    ".xans-product-detail",
    ".xans-product-detaildesign",
    ".xans-product-additional",
    "#productDetail",  # 테마별 대비
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

    # 본문 컨테이너를 못 찾으면(테마 이슈) -> 안전하게 전체에서 찾되, 마지막 fallback
    scope = container if container else soup

    urls = []
    for img in scope.select("img"):
        src = (img.get("src") or img.get("data-src") or img.get("data-original") or "").strip()
        if not src:
            continue
        full = urljoin(page_url, src)

        # data URI 같은 거 제외
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
st.title("상세페이지 썸네일 생성기 — URL은 '본문 상세이미지'만 추출")
st.caption("URL 입력 시 본문(상세영역) 이미지에서만 추출 → 450×633 여백 없이 중앙 크롭")

with st.expander("고급 옵션", expanded=False):
    max_images = st.slider("상세영역에서 수집할 최대 이미지 수", 50, 600, 250, step=50)
    st.write("※ URL 입력: 본문 상세영역(#prdDetail 등) 내부 img만 수집합니다.")

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
        "상세페이지 이미지 업로드 (a.jpg 형태, 여러 장 가능)",
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
        img = img.resize((TARGET_W, TARGET_H), Image.LANCZOS)
        fixed.append((name, img))

    st.success(f"총 {len(fixed)}장 생성 완료 (모두 {TARGET_W}×{TARGET_H}, 여백 0)")
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
