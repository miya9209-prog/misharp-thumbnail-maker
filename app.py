import io
import zipfile
import numpy as np
import streamlit as st
import requests
from bs4 import BeautifulSoup
from PIL import Image
from urllib.parse import urljoin

TARGET_W, TARGET_H = 450, 633
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

# ==================================================
# 1. 상세페이지 이미지 분해 (핵심 로직)
# ==================================================
def split_detail_image_by_white_rows(
    pil_img,
    white_thr=245,
    white_ratio=0.98,
    min_gap=60,
    min_cut_height=300,
):
    gray = pil_img.convert("L")
    arr = np.array(gray)
    h, w = arr.shape
    row_white_ratio = (arr > white_thr).mean(axis=1)

    gaps, cuts = [], []
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

    prev = 0
    for g1, g2 in gaps:
        if g1 - prev >= min_cut_height:
            cuts.append((prev, g1))
        prev = g2

    if h - prev >= min_cut_height:
        cuts.append((prev, h))

    return cuts


def trim_all_whitespace(pil_img, thr=18):
    arr = np.array(pil_img.convert("RGB"))
    h, w = arr.shape[:2]
    bg = np.median([arr[0,0], arr[0,w-1], arr[h-1,0], arr[h-1,w-1]], axis=0)
    diff = np.sqrt(((arr - bg) ** 2).sum(axis=2))
    mask = diff > thr
    if not mask.any():
        return pil_img
    ys, xs = np.where(mask)
    return pil_img.crop((xs.min(), ys.min(), xs.max()+1, ys.max()+1))


def resize_to_450x633_no_crop(pil_img):
    w, h = pil_img.size
    scale = min(TARGET_W / w, TARGET_H / h)
    return pil_img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)


def process_long_image(pil_img, prefix):
    outputs = []
    segments = split_detail_image_by_white_rows(pil_img)

    for idx, (y1, y2) in enumerate(segments, start=1):
        cut = pil_img.crop((0, y1, pil_img.width, y2))
        cut = trim_all_whitespace(cut)
        cut = resize_to_450x633_no_crop(cut)
        outputs.append((f"{prefix}_{idx:02d}_450x633.jpg", cut))

    return outputs

# ==================================================
# 2. URL에서 이미지 수집
# ==================================================
def extract_images_from_page(url):
    html = requests.get(url, headers=HEADERS, timeout=20).text
    soup = BeautifulSoup(html, "lxml")
    imgs = []

    for img in soup.select("img"):
        src = img.get("src") or img.get("data-src")
        if src and src.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            imgs.append(urljoin(url, src))

    return list(dict.fromkeys(imgs))


def download_image(url):
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")

# ==================================================
# UI
# ==================================================
st.set_page_config(layout="wide")
st.title("상세페이지 썸네일 생성기 (URL / 이미지주소 / 업로드)")
st.caption("상세페이지 이미지(a.jpg)를 상품컷 단위로 분해 → 450×633 생성")

tab1, tab2, tab3 = st.tabs(
    ["① 상세페이지 URL", "② 이미지 주소(URL)", "③ 이미지 업로드"]
)

all_outputs = []

# -------------------- ① 상세페이지 URL --------------------
with tab1:
    page_url = st.text_input("상세페이지 URL 입력")
    if st.button("URL에서 생성", key="url_go"):
        urls = extract_images_from_page(page_url)
        for i, img_url in enumerate(urls, start=1):
            pil = download_image(img_url)
            all_outputs += process_long_image(pil, f"url{i}")

# -------------------- ② 이미지 주소 --------------------
with tab2:
    url_text = st.text_area("이미지 주소(URL) 여러 개 (줄바꿈)")
    if st.button("이미지 주소로 생성", key="imgurl_go"):
        for i, line in enumerate(url_text.splitlines(), start=1):
            if line.strip():
                pil = download_image(line.strip())
                all_outputs += process_long_image(pil, f"img{i}")

# -------------------- ③ 업로드 --------------------
with tab3:
    uploads = st.file_uploader(
        "상세페이지 이미지 업로드",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
    )
    if uploads:
        for i, file in enumerate(uploads, start=1):
            pil = Image.open(file).convert("RGB")
            all_outputs += process_long_image(pil, f"up{i}")

# -------------------- 결과 --------------------
if all_outputs:
    st.success(f"총 {len(all_outputs)}장 생성 완료")
    st.image([img for _, img in all_outputs], width=180)

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
        file_name="thumb_450x633.zip",
        mime="application/zip",
    )
