import io
import os
import re
import zipfile
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse, parse_qs

import cv2
import numpy as np
import requests
import streamlit as st
from bs4 import BeautifulSoup
from PIL import Image

# ----------------------------
# Settings
# ----------------------------
TARGET_W, TARGET_H = 450, 633
TARGET_AR = TARGET_W / TARGET_H

DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
)

# Cafe24/미샵 상세 이미지가 흔히 들어가는 후보 컨테이너/셀렉터들
CAFE24_IMG_SELECTORS = [
    "#prdDetail img",
    "#prdDetailContent img",
    "#prdDetail .cont img",
    ".xans-product-detail img",
    ".xans-product-detaildesign img",
    ".xans-product-additional img",
    "#productDetail img",
    ".product-detail img",
    ".detailArea img",
]

# ----------------------------
# Data models
# ----------------------------
@dataclass
class InputImage:
    name: str
    content: bytes
    source_url: str | None = None

@dataclass
class OutputImage:
    filename: str
    image: Image.Image  # RGB

# ----------------------------
# Utility
# ----------------------------
def safe_filename(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[^\w\-.가-힣]+", "_", s)
    return s[:180] if s else "image"

def guess_product_no(page_url: str) -> str | None:
    try:
        q = parse_qs(urlparse(page_url).query)
        # cafe24: product_no=12345
        if "product_no" in q and q["product_no"]:
            return q["product_no"][0]
    except Exception:
        pass
    return None

def fetch_html(url: str) -> str:
    headers = {"User-Agent": DEFAULT_UA}
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    return r.text

def extract_title_from_html(html: str) -> str | None:
    soup = BeautifulSoup(html, "lxml")
    # 1) og:title 우선
    og = soup.select_one('meta[property="og:title"]')
    if og and og.get("content"):
        return og["content"].strip()
    # 2) title tag
    if soup.title and soup.title.text:
        t = soup.title.text.strip()
        # 흔한 노이즈 제거
        t = re.sub(r"\s*-\s*[^-]{1,40}$", "", t)
        return t.strip() or None
    return None

def is_probably_icon_by_url(u: str) -> bool:
    u = u.lower()
    bad_keywords = [
        "logo", "icon", "btn", "button", "sprite", "common", "share",
        "kakao", "facebook", "insta", "naver", "cscenter", "top", "bottom"
    ]
    return any(k in u for k in bad_keywords)

def download_image(url: str) -> InputImage | None:
    try:
        headers = {"User-Agent": DEFAULT_UA, "Referer": url}
        r = requests.get(url, headers=headers, timeout=25)
        r.raise_for_status()
        ctype = (r.headers.get("Content-Type") or "").lower()
        if "image" not in ctype:
            return None

        path = urlparse(url).path
        base = os.path.basename(path) or "image"
        if "." not in base:
            # content-type
            if "png" in ctype:
                base += ".png"
            elif "webp" in ctype:
                base += ".webp"
            else:
                base += ".jpg"

        return InputImage(name=base, content=r.content, source_url=url)
    except Exception:
        return None

# ----------------------------
# Cafe24-oriented extraction
# ----------------------------
def extract_detail_image_urls(page_url: str, html: str, max_images: int = 120) -> list[str]:
    soup = BeautifulSoup(html, "lxml")

    # 1) Cafe24 후보 셀렉터에서 먼저 수집
    urls: list[str] = []
    for sel in CAFE24_IMG_SELECTORS:
        for img in soup.select(sel):
            src = (img.get("src") or img.get("data-src") or img.get("data-original") or "").strip()
            if not src:
                continue
            abs_u = urljoin(page_url, src)
            urls.append(abs_u)

    # 2) 그래도 없으면 전체 img에서 수집
    if not urls:
        for img in soup.select("img"):
            src = (img.get("src") or img.get("data-src") or img.get("data-original") or "").strip()
            if not src:
                continue
            abs_u = urljoin(page_url, src)
            urls.append(abs_u)

    # 중복 제거
    dedup = []
    seen = set()
    for u in urls:
        if u not in seen:
            seen.add(u)
            dedup.append(u)

    return dedup[:max_images]

def filter_detail_images(pils: list[InputImage], min_w: int, min_h: int, drop_sizecharts: bool) -> list[InputImage]:
    kept: list[InputImage] = []

    for it in pils:
        if it.source_url and is_probably_icon_by_url(it.source_url):
            continue

        try:
            img = Image.open(io.BytesIO(it.content))
            w, h = img.size

            # 너무 작은 것(아이콘/버튼/배너류) 제거
            if w < min_w or h < min_h:
                continue

            # 극단적 가로배너/세로선 같은 것 제거
            ar = w / max(1, h)
            if ar > 4.5 or ar < 0.18:
                continue

            # 사이즈표 같은 이미지 흔히 "size" "chart" 키워드 포함
            if drop_sizecharts and it.source_url:
                lu = it.source_url.lower()
                if any(k in lu for k in ["size", "chart", "guide", "measure", "cm", "inch"]):
                    continue

            kept.append(it)
        except Exception:
            continue

    return kept

# ----------------------------
# Subject detection: light (saliency/contour)
# ----------------------------
def detect_subject_bbox_light(bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    h, w = bgr.shape[:2]
    if h < 40 or w < 40:
        return None

    try:
        sal = cv2.saliency.StaticSaliencyFineGrained_create()
        ok, sal_map = sal.computeSaliency(bgr)
        if not ok:
            return None
        sal_map = (sal_map * 255).astype(np.uint8)
    except Exception:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        sal_map = cv2.Canny(gray, 60, 160)

    _, th = cv2.threshold(sal_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((9, 9), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < (h * w) * 0.02:
        return None

    x, y, ww, hh = cv2.boundingRect(cnt)
    x1, y1, x2, y2 = x, y, x + ww, y + hh

    pad = int(0.06 * max(w, h))
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    return (x1, y1, x2, y2)

# ----------------------------
# Optional YOLO
# ----------------------------
def load_yolo_if_enabled(enable: bool):
    if not enable:
        return None
    try:
        from ultralytics import YOLO
        return YOLO("yolov8n.pt")
    except Exception:
        return None

def detect_subject_bbox_yolo(bgr: np.ndarray, yolo_model) -> tuple[int, int, int, int] | None:
    if yolo_model is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    results = yolo_model.predict(source=rgb, verbose=False)
    if not results:
        return None
    r0 = results[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        return None

    boxes = r0.boxes.xyxy.cpu().numpy()
    confs = r0.boxes.conf.cpu().numpy()

    best = None
    best_score = -1
    for (x1, y1, x2, y2), c in zip(boxes, confs):
        area = max(0, (x2 - x1)) * max(0, (y2 - y1))
        score = float(area) * float(c)
        if score > best_score:
            best_score = score
            best = (int(x1), int(y1), int(x2), int(y2))
    return best

# ----------------------------
# Crop logic
# ----------------------------
def expand_to_aspect(x1, y1, x2, y2, img_w, img_h, target_ar):
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0 or bh <= 0:
        return (0, 0, img_w, img_h)

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    current_ar = bw / bh

    if current_ar < target_ar:
        new_bw = bh * target_ar
        new_bh = bh
    else:
        new_bh = bw / target_ar
        new_bw = bw

    nx1 = int(round(cx - new_bw / 2))
    nx2 = int(round(cx + new_bw / 2))
    ny1 = int(round(cy - new_bh / 2))
    ny2 = int(round(cy + new_bh / 2))

    if nx1 < 0:
        nx2 -= nx1
        nx1 = 0
    if ny1 < 0:
        ny2 -= ny1
        ny1 = 0
    if nx2 > img_w:
        diff = nx2 - img_w
        nx1 -= diff
        nx2 = img_w
    if ny2 > img_h:
        diff = ny2 - img_h
        ny1 -= diff
        ny2 = img_h

    nx1 = max(0, nx1)
    ny1 = max(0, ny1)
    nx2 = min(img_w, nx2)
    ny2 = min(img_h, ny2)
    return nx1, ny1, nx2, ny2

def center_crop_to_aspect(img: Image.Image, target_ar: float) -> Image.Image:
    w, h = img.size
    if (w / h) < target_ar:
        new_h = int(w / target_ar)
        top = max(0, (h - new_h) // 2)
        return img.crop((0, top, w, top + new_h))
    else:
        new_w = int(h * target_ar)
        left = max(0, (w - new_w) // 2)
        return img.crop((left, 0, left + new_w, h))

def crop_and_resize(pil_img: Image.Image, bbox: tuple[int, int, int, int] | None) -> Image.Image:
    img = pil_img.convert("RGB")
    w, h = img.size

    if bbox is None:
        crop = center_crop_to_aspect(img, TARGET_AR)
    else:
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = expand_to_aspect(x1, y1, x2, y2, w, h, TARGET_AR)
        crop = img.crop((x1, y1, x2, y2))

    return crop.resize((TARGET_W, TARGET_H), Image.LANCZOS)

# ----------------------------
# Filename rules
# ----------------------------
def build_output_filename(
    product_no: str | None,
    product_title: str | None,
    index: int,
    scheme: str
) -> str:
    # scheme:
    # 1) productno_index
    # 2) title_index
    # 3) productno_title_index
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

# ----------------------------
# UI (실무형: 최소 버튼)
# ----------------------------
st.set_page_config(page_title="미샵 상세 썸네일 생성기", layout="wide")
st.title("미샵 상세페이지 썸네일 생성기 (450×633)")

st.caption("URL 붙여넣기 → [썸네일 만들기] → ZIP 다운로드. (업로드도 가능)")

# 핵심 입력 2개만 노출
url = st.text_input("상세페이지 URL", placeholder="https://.../product/detail.html?product_no=XXXXX")
uploads = st.file_uploader("또는 이미지 업로드(여러 장 가능)", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True)

# Advanced (숨김)
with st.expander("고급 옵션(보통은 안 건드려도 됨)"):
    use_yolo = st.checkbox("YOLO 사용(정확도↑, 첫 실행 시 모델 다운로드 필요할 수 있음)", value=False)
    max_images = st.slider("URL 모드: 최대 이미지 수", 20, 250, 120, step=10)
    min_w = st.slider("작은 이미지 제외: 최소 가로(px)", 200, 1200, 600, step=50)
    min_h = st.slider("작은 이미지 제외: 최소 세로(px)", 200, 2000, 600, step=50)
    drop_sizecharts = st.checkbox("사이즈표/차트 이미지 제외(키워드 기반)", value=True)
    filename_scheme = st.selectbox(
        "파일명 규칙",
        ["상품번호_상품명_순번", "상품번호_순번", "상품명_순번"],
        index=0
    )

make_btn = st.button("썸네일 만들기", type="primary")

if make_btn:
    if not url and not uploads:
        st.error("URL 또는 이미지 업로드 중 하나는 필요해요.")
        st.stop()

    yolo_model = load_yolo_if_enabled(use_yolo)
    if use_yolo and yolo_model is None:
        st.warning("YOLO 로딩 실패 → 기본 감지로 진행합니다.")

    inputs: list[InputImage] = []
    product_no = None
    product_title = None

    # 1) URL 모드
    if url:
        product_no = guess_product_no(url)
        with st.spinner("상세페이지 분석 중…"):
            try:
                html = fetch_html(url)
                product_title = extract_title_from_html(html)
                img_urls = extract_detail_image_urls(url, html, max_images=max_images)
            except Exception as e:
                st.error(f"URL 읽기 실패: {e}")
                st.stop()

        # 다운로드
        with st.spinner("상세 이미지 다운로드 중…"):
            for u in img_urls:
                it = download_image(u)
                if it:
                    inputs.append(it)

        if not inputs:
            st.error("다운로드 가능한 상세 이미지가 없어요. (차단/권한/DOM 구조 차이)")
            st.stop()

        inputs = filter_detail_images(inputs, min_w=min_w, min_h=min_h, drop_sizecharts=drop_sizecharts)

    # 2) 업로드 모드(추가/대체)
    if uploads:
        for f in uploads:
            inputs.append(InputImage(name=f.name, content=f.read(), source_url=None))

    if not inputs:
        st.error("필터링 후 남은 이미지가 없어요. 최소 가로/세로 기준을 낮춰보세요.")
        st.stop()

    # 처리
    outputs: list[OutputImage] = []
    previews: list[Image.Image] = []

    with st.spinner("피사체 감지 → 450×633 크롭 생성 중…"):
        for i, item in enumerate(inputs, start=1):
            try:
                pil = Image.open(io.BytesIO(item.content)).convert("RGB")
                bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

                bbox = None
                # YOLO 우선
                if yolo_model is not None:
                    bbox = detect_subject_bbox_yolo(bgr, yolo_model)
                if bbox is None:
                    bbox = detect_subject_bbox_light(bgr)

                out = crop_and_resize(pil, bbox)

                out_name = build_output_filename(
                    product_no=product_no,
                    product_title=product_title,
                    index=i,
                    scheme=filename_scheme
                )

                outputs.append(OutputImage(filename=out_name, image=out))
                if len(previews) < 12:
                    previews.append(out)
            except Exception:
                continue

    if not outputs:
        st.error("생성 결과가 없어요. (이미지 손상/형식 문제 가능)")
        st.stop()

    st.success(f"완료: {len(outputs)}개 생성")

    # 미리보기
    st.subheader("미리보기(최대 12개)")
    st.image(previews, width=180)

    # ZIP 생성
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
        file_name=f"misharp_thumbs_{TARGET_W}x{TARGET_H}.zip",
        mime="application/zip",
    )

    # 간단 로그
    st.caption(
        f"파일명 규칙: {filename_scheme} / 상품번호: {product_no or '-'} / 상품명: {product_title or '-'}"
    )
