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
# 1) Split long detail image into pieces by "horizontal white gaps"
# =========================
def split_detail_image_by_white_rows(
    pil_img: Image.Image,
    white_thr: int = 245,
    white_ratio: float = 0.985,
    min_gap: int = 70,
    min_cut_height: int = 220,
):
    """
    긴 상세페이지 이미지에서 '가로 흰 띠(여백)'를 찾아 컷 분리.
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
            if in_gap:
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
# 2) Strong whitespace trim (true cropping)
# =========================
def trim_edge_whitespace(
    pil_img: Image.Image,
    white_thr: int = 245,
    white_ratio: float = 0.985,
    min_run: int = 6,
):
    """
    이미지 가장자리에서 '거의 흰색'인 행/열을 반복적으로 잘라냄.
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

        # 너무 과한 트림 방지
        if (bottom - top) < 80 or (right - left) < 80:
            return img

        arr = arr[top : bottom + 1, left : right + 1, :]
        h, w = arr.shape[:2]
        top, bottom, left, right = 0, h - 1, 0, w - 1

    return Image.fromarray(arr)


# =========================
# 3) Foreground bbox (subject) detection by background color difference
# =========================
def foreground_bbox(
    pil_img: Image.Image,
    diff_thr: int = 22,
    margin: int = 6,
):
    """
    코너 배경색(대개 흰/연회색)과의 색 차이로 전경 마스크를 만든 뒤 bbox 계산.
    """
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
        # 전경 검출 실패면 전체를 전경으로
        return (0, 0, w - 1, h - 1)

    ys, xs = np.where(mask)
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w - 1, x2 + margin)
    y2 = min(h - 1, y2 + margin)

    return (x1, y1, x2, y2)


# =========================
# 4) Crop to target aspect around subject (NO distortion, NO blur, NO padding)
# =========================
def crop_to_aspect_keep_subject(
    pil_img: Image.Image,
    target_ar: float,
    bbox,
    extra_margin_ratio: float = 0.06,
):
    """
    - target_ar(450/633)에 맞게 크롭
    - 중심은 피사체 bbox 중심
    - 가능하면 bbox 전체가 들어가도록 크롭 크기 결정
    - 결과는 "패딩 없이" 크롭만 수행 (흰 여백 없음)
    """
    img = pil_img.convert("RGB")
    W, H = img.size
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1 + 1)
    bh = max(1, y2 - y1 + 1)

    # bbox에 약간 여유
    mx = int(bw * extra_margin_ratio)
    my = int(bh * extra_margin_ratio)
    bx1 = max(0, x1 - mx)
    by1 = max(0, y1 - my)
    bx2 = min(W - 1, x2 + mx)
    by2 = min(H - 1, y2 + my)

    bw = bx2 - bx1 + 1
    bh = by2 - by1 + 1

    # bbox를 포함하는 최소 crop 크기(비율 고정)
    # crop_w/crop_h = target_ar
    crop_w = max(bw, int(target_ar * bh))
    crop_h = int(round(crop_w / target_ar))
    if crop_h < bh:
        crop_h = bh
        crop_w = int(round(target_ar * crop_h))

    # 이미지보다 커지면 가능한 최대치로 줄이기 (그래도 bbox 못 담으면 현실적으로 불가)
    if crop_w > W:
        crop_w = W
        crop_h = int(round(crop_w / target_ar))
    if crop_h > H:
        crop_h = H
        crop_w = int(round(target_ar * crop_h))

    # 중심은 bbox 중심
    cx = (bx1 + bx2) / 2.0
    cy = (by1 + by2) / 2.0

    left = int(round(cx - crop_w / 2))
    top = int(round(cy - crop_h / 2))

    # 범위 보정
    left = max(0, min(left, W - crop_w))
    top = max(0, min(top, H - crop_h))

    return img.crop((left, top, left + crop_w, top + crop_h))


def make_thumb_450x633(pil_img: Image.Image):
    """
    최종 파이프라인:
    1) 가장자리 흰 여백 강제 트림
    2) 전경 bbox 탐지
    3) 피사체 중심으로 450:633 비율 크롭(패딩/블러/왜곡 없음)
    4) 450x633 리사이즈
    """
    cut = trim_edge_whitespace(pil_img)
    bbox = foreground_bbox(cut)
    cropped = crop_to_aspect_keep_subject(cut, TARGET_AR, bbox)
    return cropped.resize((TARGET_W, TARGET_H), Image.LANCZOS)


# =========================
# Page image extraction
# =========================
def extract_image_urls_from_page(page_url: str, max_images: int = 250) -> list[str]:
    html = requests.get(page_url, headers=HEADERS, timeout=25).text
    soup = BeautifulSoup(html, "lxml")

    selectors = [
        "#prdDetail img",
        "#prdDetailContent img",
        ".xans-product-detail img",
        ".xans-product-detaildesign img",
        ".xans-product-additional img",
        "img",
    ]

    urls = []
    for sel in selectors:
        for img in soup.select(sel):
            src = (img.get("src") or img.get("data-src") or img.get("data-original") or "").strip()
            if not src:
                continue
            full = urljoin(page_url, src)
            if full not in urls:
                urls.append(full)
            if len(urls) >= max_images:
                return urls

    return urls


# =========================
# Processing for one long detail image
# =========================
def process_long_image(pil_img: Image.Image, prefix: str):
    outputs = []
    segments = split_detail_image_by_white_rows(pil_img)

    # 분할이 실패하면 전체를 1컷으로 처리
    if not segments:
        segments = [(0, pil_img.height)]

    for idx, (y1, y2) in enumerate(segments, start=1):
        piece = pil_img.crop((0, y1, pil_img.width, y2))
        thumb = make_thumb_450x633(piece)  # <-- 여기서 “여백 없이 + 왜곡 없이 + 450x633” 완성
        outputs.append((f"{prefix}_{idx:02d}_{TARGET_W}x{TARGET_H}.jpg", thumb))

    return outputs


# =========================
# Streamlit UI
# =========================
st.set_page_config(layout="wide")
st.title("상세페이지 썸네일 생성기 (정확 크롭 버전) — 450×633")
st.caption("블러/배경합성 없이, 피사체 중심으로 '여백 없이' 450×633 크롭합니다. (왜곡 없음)")

with st.expander("고급 옵션 (기본값 권장)", expanded=False):
    max_images = st.slider("상세페이지에서 수집할 최대 이미지 수", 50, 600, 250, step=50)
    st.write("※ 이 버전은 '블러 배경'을 사용하지 않습니다.")
    st.write("※ 결과는 항상 450×633이며, 흰 여백은 크롭으로 제거합니다.")

tab1, tab2, tab3 = st.tabs(["① 상세페이지 URL", "② 이미지 주소(URL)", "③ 이미지 업로드"])

all_outputs = []

# ① 상세페이지 URL
with tab1:
    page_url = st.text_input("상세페이지 URL", placeholder="https://.../product/detail.html?product_no=28461")
    if st.button("URL에서 이미지 수집 → 생성", type="primary", key="go1"):
        if not page_url.strip():
            st.error("상세페이지 URL을 입력해주세요.")
        else:
            with st.spinner("이미지 URL 수집 중…"):
                urls = extract_image_urls_from_page(page_url.strip(), max_images=max_images)

            if not urls:
                st.error("이미지 URL을 찾지 못했습니다.")
            else:
                ok = 0
                with st.spinner(f"다운로드 및 처리 중… ({len(urls)}개 후보)"):
                    for i, u in enumerate(urls, start=1):
                        try:
                            pil = download_image(u)
                            # 상세페이지에 있는 긴 이미지든, 단일 컷 이미지든 모두 처리
                            all_outputs += process_long_image(pil, f"url{i:03d}")
                            ok += 1
                        except Exception:
                            continue
                if ok == 0:
                    st.error("처리 가능한 이미지가 없습니다. (접근 제한/차단 가능)")

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
                        all_outputs += process_long_image(pil, f"img{i:03d}")
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
                    all_outputs += process_long_image(pil, f"up{i:03d}_{base}")
                except Exception:
                    continue

# 결과 출력
if all_outputs:
    # 최종 사이즈 강제 보장
    fixed = []
    for name, img in all_outputs:
        img = img.resize((TARGET_W, TARGET_H), Image.LANCZOS)
        fixed.append((name, img))

    st.success(f"총 {len(fixed)}장 생성 완료 (모두 {TARGET_W}×{TARGET_H})")
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
