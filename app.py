import io
import os
import re
import zipfile
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse, parse_qs

import numpy as np
import requests
import streamlit as st
from bs4 import BeautifulSoup
from PIL import Image

# =========================
# Config
# =========================
TARGET_W, TARGET_H = 450, 633
TARGET_AR = TARGET_W / TARGET_H

DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
)

# Cafe24 상세영역 후보 셀렉터 (미샵 포함)
CAFE24_IMG_SELECTORS = [
    "#prdDetail img",
    "#prdDetailContent img",
    ".xans-product-detail img",
    ".xans-product-detaildesign img",
    ".xans-product-additional img",
    "#productDetail img",
    ".product-detail img",
    ".detailArea img",
]

# URL 키워드 기반 제외(아이콘/로고/버튼 등)
BAD_URL_KEYWORDS = [
    "logo", "icon", "btn", "button", "sprite", "common", "share",
    "kakao", "facebook", "insta", "naver", "top", "bottom", "header", "footer"
]

# =========================
# Models
# =========================
@dataclass
class InputImage:
    name: str
    content: bytes
    source_url: str | None = None

@dataclass
class OutputImage:
    filename: str
    image: Image.Image  # RGB

# =========================
# Utils
# =========================
def safe_filename(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^\w\-.가-힣]+", "_", s)
    return s[:180] if s else "item"

def guess_product_no(page_url: str) -> str | None:
    try:
        q = parse_qs(urlparse(page_url).query)
        if "product_no" in q and q["product_no"]:
            return q["product_no"][0]
    except Exception:
        pass
    return None

def fetch_html(url: str) -> str:
    headers = {"User-Agent": DEFAULT_UA}
    r = requests.get(url, headers=headers, timeout=25)
    r.raise_for_status()
    return r.text

def extract_title_from_html(html: str) -> str | None:
    soup = BeautifulSoup(html, "lxml")
    og = soup.select_one('meta[property="og:title"]')
    if og and og.get("content"):
        return og["content"].strip()
    if soup.title and soup.title.text:
        t = soup.title.text.strip()
        t = re.sub(r"\s*-\s*[^-]{1,40}$", "", t).strip()
        return t or None
    return None

def pick_best_from_srcset(srcset: str) -> str | None:
    # "url 300w, url2 720w" -> 가장 큰 w 선택
    if not srcset:
        return None
    parts = [p.strip() for p in srcset.split(",") if p.strip()]
    best_url = None
    best_w = -1
    for p in parts:
        seg = p.split()
        u = seg[0]
        w = 0
        if len(seg) > 1 and seg[1].endswith("w"):
            try:
                w = int(seg[1][:-1])
            except Exception:
                w = 0
        if w > best_w:
            best_w = w
            best_url = u
    return best_url

def is_probably_bad_url(u: str) -> bool:
    if not u:
        return False
    lu = u.lower()
    return any(k in lu for k in BAD_URL_KEYWORDS)

def download_image(url: str) -> InputImage | None:
    try:
        headers = {"User-Agent": DEFAULT_UA, "Referer": url}
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        ctype = (r.headers.get("Content-Type") or "").lower()
        if "image" not in ctype:
            return None

        path = urlparse(url).path
        base = os.path.basename(path) or "image"
        if "." not in base:
            if "png" in ctype:
                base += ".png"
            elif "webp" in ctype:
                base += ".webp"
            else:
                base += ".jpg"

        return InputImage(name=base, content=r.content, source_url=url)
    except Exception:
        return None

def extract_image_urls_from_page(page_url: str, html: str, max_images: int = 300) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    urls: list[str] = []

    # 1) 상세영역 후보 셀렉터 우선
    for sel in CAFE24_IMG_SELECTORS:
        nodes = soup.select(sel)
        for img in nodes:
            src = (img.get("src") or img.get("data-src") or img.get("data-original") or "").strip()
            srcset = (img.get("srcset") or img.get("data-srcset") or "").strip()
            best = pick_best_from_srcset(srcset) or src
            if best:
                urls.append(urljoin(page_url, best))

    # 2) 그래도 부족하면 전체 img
    if not urls:
        for img in soup.select("img"):
            src = (img.get("src") or img.get("data-src") or img.get("data-original") or "").strip()
            srcset = (img.get("srcset") or img.get("data-srcset") or "").strip()
            best = pick_best_from_srcset(srcset) or src
            if best:
                urls.append(urljoin(page_url, best))

    # 중복 제거
    dedup = []
    seen = set()
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            dedup.append(u)

    return dedup[:max_images]

# =========================
# 핵심: "흰 여백 제거" + "패딩 없이 450x633"
# =========================
def trim_uniform_border(pil_img: Image.Image, thr: int = 18, margin: int = 4) -> Image.Image:
    """
    코너 배경색(흰/연회색) 기준으로, 배경과 충분히 다른 영역만 남기고 크롭합니다.
    thr가 낮을수록 더 적극적으로 자르고, 높을수록 덜 자릅니다.
    """
    img = pil_img.convert("RGB")
    arr = np.array(img)
    h, w = arr.shape[:2]
    if h < 10 or w < 10:
        return img

    # 코너 4점의 중앙값으로 배경색 추정
    corners = np.array([arr[0,0], arr[0,w-1], arr[h-1,0], arr[h-1,w-1]], dtype=np.int16)
    bg = np.median(corners, axis=0)

    dist = np.sqrt(((arr.astype(np.int16) - bg) ** 2).sum(axis=2))
    mask = dist > thr

    if not mask.any():
        return img

    ys, xs = np.where(mask)
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w - 1, x2 + margin)
    y2 = min(h - 1, y2 + margin)

    return img.crop((x1, y1, x2 + 1, y2 + 1))

def crop_to_aspect_no_padding(pil_img: Image.Image, target_ar: float) -> Image.Image:
    """
    패딩(여백) 없이 목표 비율로 중앙 크롭합니다.
    """
    img = pil_img.convert("RGB")
    w, h = img.size
    if w < 2 or h < 2:
        return img

    cur_ar = w / h
    if cur_ar > target_ar:
        new_w = int(h * target_ar)
        left = (w - new_w) // 2
        return img.crop((left, 0, left + new_w, h))
    else:
        new_h = int(w / target_ar)
        top = (h - new_h) // 2
        return img.crop((0, top, w, top + new_h))

def make_thumb_450x633(pil_img: Image.Image, trim_thr: int = 18, trim_margin: int = 4) -> Image.Image:
    """
    1) 흰/연회색 여백 제거(trim)
    2) 450x633 비율로 패딩 없이 크롭
    3) 450x633 리사이즈
    """
    trimmed = trim_uniform_border(pil_img, thr=trim_thr, margin=trim_margin)
    cropped = crop_to_aspect_no_padding(trimmed, TARGET_AR)
    return cropped.resize((TARGET_W, TARGET_H), Image.LANCZOS)

# =========================
# "상품컷 위주" 필터(휴리스틱)
# =========================
def is_probably_text_or_chart(pil_img: Image.Image) -> bool:
    """
    사이즈표/텍스트 안내컷을 걸러내기 위한 간단 휴리스틱.
    - 흰 배경 비중이 높고
    - 얇은 선/글자처럼 작은 변화가 많으면(text-like)
    """
    img = pil_img.convert("L")
    arr = np.array(img).astype(np.float32)
    h, w = arr.shape[:2]
    if h < 120 or w < 120:
        return True  # 너무 작은 건 대부분 아이콘/텍스트류

    # 매우 밝은 픽셀 비율(흰 배경)
    white_ratio = float((arr > 245).mean())

    # 에지/텍스트 느낌: 인접 차이가 큰 픽셀 비율
    dx = np.abs(arr[:, 1:] - arr[:, :-1])
    dy = np.abs(arr[1:, :] - arr[:-1, :])
    edge_ratio = float(((dx > 25).mean() + (dy > 25).mean()) / 2.0)

    # 어두운 픽셀(글자/선) 비율
    dark_ratio = float((arr < 120).mean())

    # 극단 비율(배너/줄 같은 것)
    ar = w / h
    if ar > 4.8 or ar < 0.18:
        return True

    # 판단 규칙(경험적)
    # - white_ratio 높고(>0.55)
    # - dark_ratio 낮은데(0.005~0.22 정도)
    # - edge_ratio가 상대적으로 높으면 텍스트/표 가능성↑
    if white_ratio > 0.55 and dark_ratio < 0.22 and edge_ratio > 0.10:
        return True

    # 매우 흰데 글자 거의 없는 큰 공백 이미지도 제외(안내 여백컷)
    if white_ratio > 0.80 and dark_ratio < 0.03:
        return True

    return False

def build_output_filename(product_no: str | None, product_title: str | None, index: int, scheme: str) -> str:
    idx = f"{index:02d}"
    p = safe_filename(product_no or "product")
    t = safe_filename(product_title or "item")

    if scheme == "상품번호_순번":
        base = f"{p}_{idx}"
    elif scheme == "상품명_순번":
        base = f"{t}_{idx}"
    else:
        base = f"{p}_{t}_{idx}"

    return f"{base}_thumb_{TARGET_W}x{TARGET_H}.jpg"

# =========================
# UI
# =========================
st.set_page_config(page_title="미샵 썸네일 생성기 450x633", layout="wide")
st.title("상세페이지 썸네일 이미지 생성기 (450×633, 흰 여백 제거)")

st.caption("입력 3가지: (1) 상세페이지 URL (2) 이미지 URL 목록 (3) 이미지 업로드 → 전체 ZIP 다운로드")

tab1, tab2, tab3 = st.tabs(["① 상세페이지 URL", "② 이미지 URL 목록", "③ 이미지 업로드"])

with st.expander("고급 옵션 (보통 기본값 그대로 사용)"):
    exclude_text = st.checkbox("상품컷 위주: 텍스트/사이즈표/아이콘 자동 제외", value=True)
    filename_scheme = st.selectbox("파일명 규칙", ["상품번호_상품명_순번", "상품번호_순번", "상품명_순번"], index=0)
    max_images = st.slider("URL에서 가져올 최대 이미지 수", 50, 600, 300, step=50)
    trim_thr = st.slider("흰 여백 제거 강도(thr)", 8, 40, 18, step=1, help="낮추면 더 과감히 여백을 자릅니다. 상품까지 잘리면 올리세요.")
    trim_margin = st.slider("트리밍 여유(margin)", 0, 30, 4, step=1)
    show_debug = st.checkbox("디버그: 제외된 이미지도 개수 표시", value=True)

def process_inputs(inputs: list[InputImage], product_no: str | None, product_title: str | None):
    kept = []
    dropped = []

    # 1) 1차 URL 키워드/크기 기준 (너무 작은 건 제외)
    for it in inputs:
        try:
            pil = Image.open(io.BytesIO(it.content)).convert("RGB")
            w, h = pil.size
            if it.source_url and is_probably_bad_url(it.source_url):
                dropped.append((it, "url_keyword"))
                continue
            if w < 180 or h < 180:
                dropped.append((it, "too_small"))
                continue

            if exclude_text and is_probably_text_or_chart(pil):
                dropped.append((it, "text/chart"))
                continue

            kept.append((it, pil))
        except Exception:
            dropped.append((it, "open_fail"))

    if not kept:
        st.error("남은 이미지가 없습니다. (자동 제외가 너무 강할 수 있어요) → 고급 옵션에서 '자동 제외'를 꺼보세요.")
        if show_debug and dropped:
            st.write(f"제외됨: {len(dropped)}개")
        return

    outputs: list[OutputImage] = []
    previews = []

    for idx, (it, pil) in enumerate(kept, start=1):
        out = make_thumb_450x633(pil, trim_thr=trim_thr, trim_margin=trim_margin)
        out_name = build_output_filename(product_no, product_title, idx, filename_scheme)
        outputs.append(OutputImage(filename=out_name, image=out))
        if len(previews) < 12:
            previews.append(out)

    st.success(f"완료: 입력 {len(inputs)}개 → 생성 {len(outputs)}개" + (f" (제외 {len(dropped)}개)" if show_debug else ""))
    st.subheader("미리보기 (최대 12개)")
    st.image(previews, width=180)

    # ZIP
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for o in outputs:
            img_bytes = io.BytesIO()
            o.image.save(img_bytes, format="JPEG", quality=92)
            zf.writestr(o.filename, img_bytes.getvalue())
    zip_buf.seek(0)

    st.download_button(
        "저장(전체 ZIP 다운로드)",
        data=zip_buf,
        file_name=f"thumbs_{TARGET_W}x{TARGET_H}.zip",
        mime="application/zip",
    )

    if show_debug:
        with st.expander("제외된 이미지 보기(사유)"):
            # 너무 많으면 상위 50개만
            for it, reason in dropped[:50]:
                st.write(f"- {it.name}  ({reason})")

# -------------------------
# Tab 1: Page URL
# -------------------------
with tab1:
    page_url = st.text_input("상세페이지 URL", placeholder="https://.../product/detail.html?product_no=28461")
    go1 = st.button("URL에서 전체 이미지 불러오기 → 썸네일 생성", type="primary", key="go1")

    if go1:
        if not page_url.strip():
            st.error("URL을 입력해주세요.")
        else:
            product_no = guess_product_no(page_url)
            with st.spinner("상세페이지 HTML 분석 중…"):
                try:
                    html = fetch_html(page_url)
                    product_title = extract_title_from_html(html)
                    urls = extract_image_urls_from_page(page_url, html, max_images=max_images)
                except Exception as e:
                    st.error(f"페이지 분석 실패: {e}")
                    st.stop()

            if not urls:
                st.error("이미지 URL을 찾지 못했습니다. (상세페이지 구조/차단 가능)")
                st.stop()

            inputs: list[InputImage] = []
            with st.spinner(f"이미지 다운로드 중… ({len(urls)}개 후보)"):
                for u in urls:
                    if is_probably_bad_url(u):
                        continue
                    it = download_image(u)
                    if it:
                        inputs.append(it)

            if not inputs:
                st.error("다운로드 가능한 이미지가 없습니다. (접근 제한/차단 가능) → '이미지 업로드' 방식으로도 사용 가능합니다.")
                st.stop()

            process_inputs(inputs, product_no=product_no, product_title=product_title)

# -------------------------
# Tab 2: Image URL list
# -------------------------
with tab2:
    st.write("여러 이미지 URL을 줄바꿈으로 붙여넣으세요. (각 줄 1개 URL)")
    url_text = st.text_area("이미지 URL 목록", height=200, placeholder="https://.../1.jpg\nhttps://.../2.jpg\n...")
    product_no2 = st.text_input("파일명용 상품번호(선택)", placeholder="예: 28461", key="pno2")
    product_title2 = st.text_input("파일명용 상품명(선택)", placeholder="예: 마이 스트라이프 기모 맨투맨", key="pt2")
    go2 = st.button("이미지 URL 다운로드 → 썸네일 생성", type="primary", key="go2")

    if go2:
        lines = [l.strip() for l in (url_text or "").splitlines() if l.strip()]
        if not lines:
            st.error("이미지 URL을 한 줄에 하나씩 넣어주세요.")
        else:
            inputs: list[InputImage] = []
            with st.spinner(f"다운로드 중… ({len(lines)}개)"):
                for u in lines:
                    it = download_image(u)
                    if it:
                        inputs.append(it)
            if not inputs:
                st.error("다운로드 실패. (URL 접근 제한/오타 가능)")
            else:
                process_inputs(inputs, product_no=product_no2 or None, product_title=product_title2 or None)

# -------------------------
# Tab 3: Upload
# -------------------------
with tab3:
    uploads = st.file_uploader("이미지 업로드(여러 장 가능)", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True)
    product_no3 = st.text_input("파일명용 상품번호(선택)", placeholder="예: 28461", key="pno3")
    product_title3 = st.text_input("파일명용 상품명(선택)", placeholder="예: 마이 스트라이프 기모 맨투맨", key="pt3")
    go3 = st.button("업로드 이미지 → 썸네일 생성", type="primary", key="go3")

    if go3:
        if not uploads:
            st.error("이미지를 업로드해주세요.")
        else:
            inputs = []
            for f in uploads:
                inputs.append(InputImage(name=f.name, content=f.read(), source_url=None))
            process_inputs(inputs, product_no=product_no3 or None, product_title=product_title3 or None)
