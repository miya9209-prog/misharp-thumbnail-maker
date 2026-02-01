import io
import re
import zipfile
from urllib.parse import urljoin, urlparse

import numpy as np
import requests
import streamlit as st
from bs4 import BeautifulSoup
from PIL import Image, ImageFilter

# =========================
# Fixed output size
# =========================
TARGET_W, TARGET_H = 450, 633

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
    )
}

# =========================
# Helpers
# =========================
def safe_name(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^\w\-.가-힣]+", "_", s)
    return s[:120] if s else "item"

def is_image_url(u: str) -> bool:
    u = (u or "").lower()
    return u.endswith((".jpg", ".jpeg", ".png", ".webp"))

def download_bytes(url: str) -> bytes:
    r = requests.get(url, headers={**HEADERS, "Referer": url}, timeout=30)
    r.raise_for_status()
    return r.content

def download_image(url: str) -> Image.Image:
    content = download_bytes(url)
    return Image.open(io.BytesIO(content)).convert("RGB")

# =========================
# 1) Split long detail image into pieces
#    by detecting "horizontal white gaps"
# =========================
def split_detail_image_by_white_rows(
    pil_img: Image.Image,
    white_thr: int = 245,
    white_ratio: float = 0.98,
    min_gap: int = 60,
    min_cut_height: int = 220,
):
    """
    긴 상세페이지 이미지(a.jpg) 안에서
    '완전히 흰 가로 여백 구간'을 찾아 컷을 분할합니다.
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
# 2) Strong whitespace trim (top/bottom/left/right)
# =========================
def trim_all_whitespace(
    pil_img: Image.Image,
    white_thr: int = 245,
    white_ratio: float = 0.985,
    min_run: int = 8,
) -> Image.Image:
    """
    상/하/좌/우 가장자리에서
    '거의 흰색'인 행/열을 반복적으로 제거합니다.
    (완전 흰 여백 제거 목적)
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
        if (bottom - top) < 60 or (right - left) < 60:
            return img

        arr = arr[top : bottom + 1, left : right + 1, :]
        h, w = arr.shape[:2]
        top, bottom, left, right = 0, h - 1, 0, w - 1

    return Image.fromarray(arr)

# =========================
# 3) Make final 450x633:
#    - No white border
#    - No subject crop (contain)
#    - Always exact size
#    - Fill leftover with blurred background (not white)
# =========================
def make_thumb_450x633_no_white_no_crop(pil_img: Image.Image) -> Image.Image:
    """
    1) 흰 여백 강제 제거
    2) 450x633으로 무조건 고정
       - 상품은 절대 안 잘리게 contain
       - 남는 영역은 흰색 금지 → 원본을 cover로 깔고 블러 처리
    """
    cut = trim_all_whitespace(pil_img)

    target_w, target_h = TARGET_W, TARGET_H
    w, h = cut.size

    # (A) Background: cover + crop + blur
    scale_cover = max(target_w / w, target_h / h)
    bg = cut.resize((int(w * scale_cover), int(h * scale_cover)), Image.LANCZOS)
    bw, bh = bg.size
    left = (bw - target_w) // 2
    top = (bh - target_h) // 2
    bg = bg.crop((left, top, left + target_w, top + target_h))
    bg = bg.filter(ImageFilter.GaussianBlur(radius=18))

    # (B) Foreground: contain (no crop)
    scale_contain = min(target_w / w, target_h / h)
    fg = cut.resize((int(w * scale_contain), int(h * scale_contain)), Image.LANCZOS)

    # (C) Composite
    canvas = bg.copy()
    fw, fh = fg.size
    px = (target_w - fw) // 2
    py = (target_h - fh) // 2
    canvas.paste(fg, (px, py))

    # Final guarantee
    if canvas.size != (target_w, target_h):
        canvas = canvas.resize((target_w, target_h), Image.LANCZOS)

    return canvas

# =========================
# Page image extraction (generic)
# =========================
def extract_image_urls_from_page(page_url: str, max_images: int = 400) -> list[str]:
    html = requests.get(page_url, headers=HEADERS, timeout=25).text
    soup = BeautifulSoup(html, "lxml")

    urls = []
    # 먼저 상세영역 후보(카페24 포함) + 일반 img
    selectors = [
        "#prdDetail img",
        "#prdDetailContent img",
        ".xans-product-detail img",
        ".xans-product-detaildesign img",
        ".xans-product-additional img",
        "img",
    ]

    for sel in selectors:
        for img in soup.select(sel):
            src = (img.get("src") or img.get("data-src") or img.get("data-original") or "").strip()
            srcset = (img.get("srcset") or img.get("data-srcset") or "").strip()

            best = None
            if srcset:
                # pick biggest width in srcset
                parts = [p.strip() for p in srcset.split(",") if p.strip()]
                best_w = -1
                for p in parts:
                    seg = p.split()
                    u = seg[0]
                    wv = 0
                    if len(seg) > 1 and seg[1].endswith("w"):
                        try:
                            wv = int(seg[1][:-1])
                        except:
                            wv = 0
                    if wv > best_w:
                        best_w = wv
                        best = u

            u = best or src
            if not u:
                continue
            full = urljoin(page_url, u)

            # 이미지 확장자 없는 경우도 있어서 완화
            if full not in urls:
                urls.append(full)

            if len(urls) >= max_images:
                return urls

    return urls

# =========================
# Core processing for one long image
# =========================
def process_long_image(pil_img: Image.Image, prefix: str) -> list[tuple[str, Image.Image]]:
    outputs = []

    # 1) split into pieces
    segments = split_detail_image_by_white_rows(pil_img)

    # fallback: if not split well, treat whole image as one piece
    if not segments:
        segments = [(0, pil_img.height)]

    for idx, (y1, y2) in enumerate(segments, start=1):
        piece = pil_img.crop((0, y1, pil_img.width, y2))
        thumb = make_thumb_450x633_no_white_no_crop(piece)

        # final hard guarantee
        thumb = thumb.resize((TARGET_W, TARGET_H), Image.LANCZOS)

        outputs.append((f"{prefix}_{idx:02d}_450x633.jpg", thumb))

    return outputs

# =========================
# Streamlit UI
# =========================
st.set_page_config(layout="wide")
st.title("상세페이지 썸네일 생성기 (URL / 이미지주소 / 업로드) - 450×633 고정")
st.caption("긴 상세페이지 이미지(a.jpg)를 상품컷 단위로 분해 → 흰 여백 제거 → 450×633 고정 ZIP 다운로드")

with st.expander("고급 옵션 (기본값 권장)", expanded=False):
    max_images = st.slider("상세페이지에서 수집할 최대 이미지 수", 50, 800, 400, step=50)
    st.write("※ 이 버전은 '피사체 잘림 금지'를 최우선으로 하며, 흰색 여백은 블러 배경으로 제거합니다.")
    st.write("※ 최종 결과물은 무조건 450×633입니다.")

tab1, tab2, tab3 = st.tabs(["① 상세페이지 URL", "② 이미지 주소(URL)", "③ 이미지 업로드"])

all_outputs: list[tuple[str, Image.Image]] = []

# -------- Tab 1: Page URL --------
with tab1:
    page_url = st.text_input("상세페이지 URL", placeholder="https://.../product/detail.html?product_no=28461")
    go1 = st.button("URL에서 이미지 수집 → 썸네일 생성", type="primary", key="go1")

    if go1:
        if not page_url.strip():
            st.error("상세페이지 URL을 입력해주세요.")
        else:
            with st.spinner("상세페이지에서 이미지 URL 수집 중…"):
                try:
                    urls = extract_image_urls_from_page(page_url.strip(), max_images=max_images)
                except Exception as e:
                    st.error(f"이미지 URL 수집 실패: {e}")
                    st.stop()

            if not urls:
                st.error("페이지에서 이미지 URL을 찾지 못했습니다.")
                st.stop()

            ok = 0
            with st.spinner(f"이미지 다운로드 및 처리 중… (후보 {len(urls)}개)"):
                for i, u in enumerate(urls, start=1):
                    try:
                        pil = download_image(u)
                        # 긴 상세페이지 형태(a.jpg)라 가정하고 처리
                        all_outputs += process_long_image(pil, f"url{i:03d}")
                        ok += 1
                    except Exception:
                        continue

            if ok == 0:
                st.error("다운로드/처리 가능한 이미지가 없습니다. (접근 제한 가능)")

# -------- Tab 2: Image URL list --------
with tab2:
    st.write("이미지 URL을 여러 줄로 붙여넣으세요. (각 줄 1개)")
    url_text = st.text_area("이미지 주소 목록", height=180, placeholder="https://.../a.jpg\nhttps://.../b.jpg\n...")
    go2 = st.button("이미지 주소로 썸네일 생성", type="primary", key="go2")

    if go2:
        lines = [l.strip() for l in (url_text or "").splitlines() if l.strip()]
        if not lines:
            st.error("이미지 주소(URL)를 한 줄에 하나씩 넣어주세요.")
        else:
            with st.spinner(f"다운로드 및 처리 중… ({len(lines)}개)"):
                for i, u in enumerate(lines, start=1):
                    try:
                        pil = download_image(u)
                        all_outputs += process_long_image(pil, f"img{i:03d}")
                    except Exception:
                        continue

# -------- Tab 3: Upload --------
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

# -------- Results --------
if all_outputs:
    # Final hard guarantee on size + filename consistency
    fixed_outputs = []
    for name, img in all_outputs:
        img = img.resize((TARGET_W, TARGET_H), Image.LANCZOS)
        fixed_outputs.append((name, img))

    st.success(f"총 {len(fixed_outputs)}장 생성 완료 (모두 {TARGET_W}×{TARGET_H})")

    st.subheader("미리보기 (일부)")
    st.image([img for _, img in fixed_outputs[:24]], width=180)

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, img in fixed_outputs:
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
