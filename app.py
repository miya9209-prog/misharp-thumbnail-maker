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
# Default output size
# =========================
DEFAULT_TARGET_W, DEFAULT_TARGET_H = 450, 633

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


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# =========================
# Background / subject mask
# =========================
def estimate_background_color(arr_rgb: np.ndarray) -> np.ndarray:
    h, w = arr_rgb.shape[:2]
    band = max(2, min(h, w) // 28)
    border = np.concatenate(
        [
            arr_rgb[:band, :, :].reshape(-1, 3),
            arr_rgb[h - band :, :, :].reshape(-1, 3),
            arr_rgb[:, :band, :].reshape(-1, 3),
            arr_rgb[:, w - band :, :].reshape(-1, 3),
        ],
        axis=0,
    ).astype(np.int16)
    return np.median(border, axis=0)


def subject_mask(arr: np.ndarray) -> np.ndarray:
    arr16 = arr.astype(np.int16)
    bg = estimate_background_color(arr16)
    diff = np.sqrt(((arr16 - bg) ** 2).sum(axis=2))
    lum = arr16.mean(axis=2)
    bg_lum = float(bg.mean())
    sat = arr16.max(axis=2) - arr16.min(axis=2)

    # 밝은 스튜디오 배경 + 어두운/컬러 피사체를 같이 잡습니다.
    mask = (diff > 22) | (np.abs(lum - bg_lum) > 16) | ((sat > 28) & (diff > 12))

    # 순백/연회색 배경은 제거합니다.
    white_bg = (arr16[:, :, 0] > 245) & (arr16[:, :, 1] > 245) & (arr16[:, :, 2] > 245)
    mask = mask & (~white_bg)
    return mask


def component_bboxes(mask: np.ndarray, min_area_ratio: float = 0.0012):
    """피사체 후보 연결 성분 bbox 목록. 텍스트 조각은 대체로 작고 납작하므로 제외합니다."""
    h, w = mask.shape
    try:
        import cv2
        m = (mask.astype(np.uint8) * 255)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        comps = []
        area_min = max(40, int(h * w * min_area_ratio))
        for lab in range(1, num):
            x, y, bw, bh, area = stats[lab]
            if area < area_min:
                continue
            comps.append((int(x), int(y), int(x + bw), int(y + bh), int(area)))
        comps.sort(key=lambda x: x[4], reverse=True)
        return comps
    except Exception:
        comp = largest_component_bbox_fallback(mask)
        return [comp] if comp else []


def largest_component_bbox_fallback(mask: np.ndarray):
    h, w = mask.shape
    rows = np.where(mask.mean(axis=1) > 0.01)[0]
    cols = np.where(mask.mean(axis=0) > 0.01)[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    return (int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1]), int(mask.sum()))


def _valid_subject_component(comp, w: int, h: int) -> bool:
    l, t, r, b, area = comp
    bw, bh = r - l, b - t
    if bw <= 0 or bh <= 0:
        return False
    area_ratio = area / float(w * h)
    box_ratio = (bw * bh) / float(w * h)
    hr = bh / float(h)
    wr = bw / float(w)
    aspect = bh / max(bw, 1)

    # 텍스트 한 줄/작은 로고/아이콘은 제외. 모델·옷걸이 상품·착장 컷은 통과.
    if area_ratio < 0.006 and box_ratio < 0.045:
        return False
    if hr < 0.20 or wr < 0.08:
        return False
    if aspect < 0.45 and hr < 0.42:
        return False
    return True


def subject_bbox(pil_img: Image.Image):
    img = pil_img.convert("RGB")
    arr = np.array(img)
    h, w = arr.shape[:2]
    mask = subject_mask(arr)
    comps = component_bboxes(mask)
    valid = [c for c in comps if _valid_subject_component(c, w, h)]

    if valid:
        # 가장 큰 피사체 1개만 사용합니다. 여러 모델/여러 상품이 같이 있으면 1개만 중심으로 잡습니다.
        left, top, right, bottom, _ = valid[0]
    else:
        # fallback도 너무 쉽게 쓰지 않습니다. 텍스트/안내컷 생성을 막는 쪽으로 보수적으로 처리합니다.
        rows = np.where(mask.mean(axis=1) > 0.012)[0]
        cols = np.where(mask.mean(axis=0) > 0.012)[0]
        if len(rows) == 0 or len(cols) == 0:
            return None
        left, right = int(cols[0]), int(cols[-1])
        top, bottom = int(rows[0]), int(rows[-1])
        bw, bh = right - left + 1, bottom - top + 1
        if bh / float(h) < 0.36 or bw / float(w) < 0.16 or (bw * bh) / float(w * h) < 0.08:
            return None

    pad_x = max(8, int((right - left + 1) * 0.075))
    pad_y = max(8, int((bottom - top + 1) * 0.065))
    left = clamp(left - pad_x, 0, w - 1)
    right = clamp(right + pad_x, 1, w)
    top = clamp(top - pad_y, 0, h - 1)
    bottom = clamp(bottom + pad_y, 1, h)
    if (right - left) < 45 or (bottom - top) < 70:
        return None
    return (left, top, right, bottom)


def has_usable_subject(pil_img: Image.Image) -> bool:
    """본문 설명/혜택/로고/원단 확대/텍스트 이미지 등 썸네일 부적합 이미지를 제외."""
    img = pil_img.convert("RGB")
    w, h = img.size
    if w < 180 or h < 180:
        return False
    arr = np.array(img)
    bbox = subject_bbox(img)
    if bbox is None:
        return False
    l, t, r, b = bbox
    bw, bh = r - l, b - t
    bbox_ratio = (bw * bh) / float(w * h)
    height_ratio = bh / float(h)
    width_ratio = bw / float(w)

    white_ratio = ((arr[:, :, 0] > 242) & (arr[:, :, 1] > 242) & (arr[:, :, 2] > 242)).mean()
    gray = arr.mean(axis=2).astype(np.float32)
    edge_density = ((np.abs(np.diff(gray, axis=1)) > 22).mean() + (np.abs(np.diff(gray, axis=0)) > 22).mean()) / 2
    sat_mean = (arr.max(axis=2) - arr.min(axis=2)).mean()

    # 흑백/회색 위주의 안내·혜택·브랜드스토리·텍스트 카드 차단.
    # 실제 상품/모델 컷은 흰 셔츠라도 피부·배경·그림자 때문에 평균 채도가 이보다 높게 나옵니다.
    if sat_mean < 1.3 and edge_density > 0.007:
        return False

    # 텍스트/혜택/브랜드스토리 컷은 대부분 흰 바탕 + 작은 조각들이고, 유효 피사체 비율이 작습니다.
    if height_ratio < 0.24 or width_ratio < 0.10 or bbox_ratio < 0.045:
        return False
    if white_ratio > 0.70 and bbox_ratio < 0.22 and edge_density > 0.022:
        return False
    if white_ratio > 0.82 and height_ratio < 0.62:
        return False

    # 중앙 피사체 없이 문구가 넓게 퍼진 컷 차단
    crop = arr[t:b, l:r]
    if crop.size == 0:
        return False
    crop_white = ((crop[:, :, 0] > 242) & (crop[:, :, 1] > 242) & (crop[:, :, 2] > 242)).mean()
    if crop_white > 0.78 and bbox_ratio < 0.35:
        return False

    # 원단/디테일 확대는 썸네일 대표컷에서 제외. 옷걸이 단품은 중앙 여백이 있어 통과합니다.
    if bbox_ratio > 0.90 and edge_density > 0.035:
        return False
    return True


# =========================
# Trimming / splitting
# =========================
def trim_edge_bands(pil_img: Image.Image, white_thr: int = 246, black_thr: int = 9):
    """가장자리의 큰 흰/검정 띠를 벡터 방식으로 빠르게 제거."""
    img = pil_img.convert("RGB")
    arr = np.array(img).astype(np.int16)
    h, w = arr.shape[:2]
    solid_ratio_thr = 0.985
    std_thr = 10.0

    row_white = (arr > white_thr).all(axis=2).mean(axis=1)
    row_black = (arr < black_thr).all(axis=2).mean(axis=1)
    row_std = arr.std(axis=1).mean(axis=1)
    row_band = ((row_white >= solid_ratio_thr) | (row_black >= solid_ratio_thr)) & (row_std <= std_thr)

    col_white = (arr > white_thr).all(axis=2).mean(axis=0)
    col_black = (arr < black_thr).all(axis=2).mean(axis=0)
    col_std = arr.std(axis=0).mean(axis=1)
    col_band = ((col_white >= solid_ratio_thr) | (col_black >= solid_ratio_thr)) & (col_std <= std_thr)

    top = 0
    while top < h - 1 and row_band[top]:
        top += 1
    bottom = h - 1
    while bottom > top and row_band[bottom]:
        bottom -= 1
    left = 0
    while left < w - 1 and col_band[left]:
        left += 1
    right = w - 1
    while right > left and col_band[right]:
        right -= 1

    if (right - left + 1) < max(160, w * 0.35) or (bottom - top + 1) < max(160, h * 0.35):
        return img
    return img.crop((left, top, right + 1, bottom + 1))

def _runs_from_bool(flags, min_len):
    runs, start = [], None
    for i, v in enumerate(flags):
        if v and start is None:
            start = i
        elif (not v) and start is not None:
            if i - start >= min_len:
                runs.append((start, i))
            start = None
    if start is not None and len(flags) - start >= min_len:
        runs.append((start, len(flags)))
    return runs


def _solid_gap_cuts(arr, axis: int, min_gap: int):
    # axis=0 horizontal row cuts, axis=1 vertical col cuts
    a = arr.astype(np.int16)
    white_thr, black_thr = 246, 9
    if axis == 0:
        white = (a > white_thr).all(axis=2).mean(axis=1)
        black = (a < black_thr).all(axis=2).mean(axis=1)
        std = a.std(axis=1).mean(axis=1)
    else:
        white = (a > white_thr).all(axis=2).mean(axis=0)
        black = (a < black_thr).all(axis=2).mean(axis=0)
        std = a.std(axis=0).mean(axis=1)
    flags = ((white > 0.965) | (black > 0.965)) & (std < 13)
    return [int((s + e) / 2) for s, e in _runs_from_bool(flags, min_gap)]


def _seam_cuts(arr, axis: int, min_piece: int):
    """붙어있는 2장 이상 사진의 경계선 탐지. 흰 여백이 없어도 색/명암 급변 라인을 잡습니다."""
    gray = arr.mean(axis=2).astype(np.float32)
    if axis == 0:
        # 인접 행의 전체 폭 평균 변화량
        score = np.abs(np.diff(gray, axis=0)).mean(axis=1)
        length = arr.shape[0]
    else:
        score = np.abs(np.diff(gray, axis=1)).mean(axis=0)
        length = arr.shape[1]
    if len(score) < min_piece * 2:
        return []
    med = float(np.median(score))
    p98 = float(np.percentile(score, 98))
    p995 = float(np.percentile(score, 99.5))
    thr = max(med * 4.0, p98 * 1.18, 10.0)
    candidates = np.where(score >= thr)[0] + 1
    cuts = []
    for c in candidates:
        if c < min_piece or length - c < min_piece:
            continue
        # 너무 가까운 경계는 가장 강한 것 하나만 유지
        if cuts and c - cuts[-1] < min_piece:
            prev = cuts[-1]
            if score[c - 1] > score[prev - 1]:
                cuts[-1] = int(c)
        else:
            cuts.append(int(c))
    # 경계가 너무 많으면 강한 것 위주로만
    if len(cuts) > 6:
        cuts = sorted(cuts, key=lambda x: score[x - 1], reverse=True)[:6]
        cuts = sorted(cuts)
    return cuts


def split_touching_images(pil_img: Image.Image, target_w: int, target_h: int):
    """상세 이미지 한 장 안에 여러 사진이 위아래/좌우로 붙은 경우 분리."""
    img = trim_edge_bands(pil_img)
    arr = np.array(img.convert("RGB"))
    h, w = arr.shape[:2]
    min_h = max(220, int(target_h * 0.55))
    min_w = max(180, int(target_w * 0.50))

    hcuts = _solid_gap_cuts(arr, axis=0, min_gap=max(8, h // 140)) + _seam_cuts(arr, axis=0, min_piece=min_h)
    vcuts = _solid_gap_cuts(arr, axis=1, min_gap=max(8, w // 140)) + _seam_cuts(arr, axis=1, min_piece=min_w)
    hcuts = sorted(set([c for c in hcuts if min_h <= c <= h - min_h]))
    vcuts = sorted(set([c for c in vcuts if min_w <= c <= w - min_w]))

    # 위아래로 길게 이어붙은 상세컷을 우선 분리합니다.
    pieces = []
    if hcuts:
        bounds = [0] + hcuts + [h]
        for y1, y2 in zip(bounds[:-1], bounds[1:]):
            if y2 - y1 >= min_h:
                pieces.append(img.crop((0, y1, w, y2)))
    elif vcuts:
        bounds = [0] + vcuts + [w]
        for x1, x2 in zip(bounds[:-1], bounds[1:]):
            if x2 - x1 >= min_w:
                pieces.append(img.crop((x1, 0, x2, h)))
    else:
        pieces = [img]

    # 과분할/무한 재귀 방지를 위해 1차 경계선 기준으로만 분리합니다.
    return pieces


# =========================
# Thumbnail generation
# =========================
def subject_center(pil_img: Image.Image):
    bbox = subject_bbox(pil_img)
    w, h = pil_img.size
    if bbox is None:
        return (w / 2.0, h / 2.0)
    l, t, r, b = bbox
    return ((l + r) / 2.0, (t + b) / 2.0)


def resize_cover_then_crop(pil_img: Image.Image, target_w: int, target_h: int, center_xy=None):
    img = pil_img.convert("RGB")
    W, H = img.size
    scale = max(target_w / W, target_h / H)
    new_w, new_h = int(round(W * scale)), int(round(H * scale))
    resized = img.resize((new_w, new_h), Image.LANCZOS)

    if center_xy is None:
        cx, cy = new_w / 2.0, new_h / 2.0
    else:
        ox, oy = center_xy
        cx, cy = ox * scale, oy * scale

    left = int(round(cx - target_w / 2.0))
    top = int(round(cy - target_h / 2.0))
    left = clamp(left, 0, max(0, new_w - target_w))
    top = clamp(top, 0, max(0, new_h - target_h))
    return resized.crop((left, top, left + target_w, top + target_h))


def edge_bleed_fix(pil_img: Image.Image, n: int = 3):
    img = pil_img.convert("RGB")
    arr = np.array(img).copy()
    h, w = arr.shape[:2]
    n = max(1, min(n, 5))
    if h <= 2 * n + 2 or w <= 2 * n + 2:
        return img
    arr[0:n, :, :] = arr[n : n + 1, :, :]
    arr[h - n : h, :, :] = arr[h - n - 1 : h - n, :, :]
    arr[:, 0:n, :] = arr[:, n : n + 1, :]
    arr[:, w - n : w, :] = arr[:, w - n - 1 : w - n, :]
    return Image.fromarray(arr)


def crop_to_single_subject(pil_img: Image.Image):
    """텍스트/혜택 영역은 버리고, 썸네일로 쓸 1개 피사체 주변만 남깁니다."""
    img = trim_edge_bands(pil_img).convert("RGB")
    bbox = subject_bbox(img)
    if bbox is None:
        return img
    w, h = img.size
    l, t, r, b = bbox
    bw, bh = r - l, b - t
    # 피사체 주변 여백은 살리되, 상단/하단 설명 텍스트가 같이 들어오지 않도록 과한 여백은 제한합니다.
    pad_x = max(10, int(bw * 0.10))
    pad_y = max(12, int(bh * 0.08))
    l = clamp(l - pad_x, 0, w - 1)
    r = clamp(r + pad_x, 1, w)
    t = clamp(t - pad_y, 0, h - 1)
    b = clamp(b + pad_y, 1, h)
    cropped = img.crop((l, t, r, b))
    return trim_edge_bands(cropped)


def make_thumbnail(pil_img: Image.Image, target_w: int, target_h: int):
    cut = crop_to_single_subject(pil_img)
    cxy = subject_center(cut)
    out = resize_cover_then_crop(cut, target_w, target_h, center_xy=cxy)
    return edge_bleed_fix(out, n=3)


# =========================
# URL extraction
# =========================
DETAIL_CONTAINER_SELECTORS = [
    "#prdDetailContent",
    "#prdDetail",
    ".xans-product-detail",
    ".xans-product-detaildesign",
    ".xans-product-additional",
    "#productDetail",
    ".cont_detail",
    ".detailArea",
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
        src = (img.get("src") or img.get("data-src") or img.get("ec-data-src") or img.get("data-original") or "").strip()
        if not src or src.startswith("data:"):
            continue
        full = urljoin(page_url, src)
        if full not in urls:
            urls.append(full)
        if len(urls) >= max_images:
            break
    return urls


# =========================
# Processing
# =========================
def process_image_any(pil_img: Image.Image, prefix: str, target_w: int, target_h: int, skip_no_subject: bool = True):
    outputs, skipped = [], []
    pieces = split_touching_images(pil_img, target_w, target_h)

    for idx, piece in enumerate(pieces, start=1):
        piece = trim_edge_bands(piece)
        if skip_no_subject and not has_usable_subject(piece):
            skipped.append((f"{prefix}_{idx:02d}", "피사체 없음/텍스트·원단·디테일 컷으로 판단"))
            continue
        thumb = make_thumbnail(piece, target_w, target_h)
        outputs.append((f"{prefix}_{idx:02d}_{target_w}x{target_h}.jpg", thumb))
    return outputs, skipped


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
      .rule-box {background:#fff7f7; border:1px solid #f1c4c4; border-radius:12px; padding:12px 14px; color:#5b1b1b; font-size:0.93rem; line-height:1.55;}
    </style>
    <div class="misharp-title-wrap">
      <div class="misharp-title">MISHARP 상세페이지 썸네일 생성기</div>
      <div class="misharp-sub">MISHARP THUMBNAIL GENERATOR V2</div>
      <div class="misharp-caption">1장=1피사체 / 흰줄·여백 제거 / 피사체 중앙 배치 / 기본 450×633 + 사용자 지정 사이즈</div>
    </div>
    <div class="rule-box">
      절대원칙: 썸네일 1개에는 피사체 1개만 남깁니다. 두 장이 붙어 있는 상세컷은 경계선을 찾아 분리하고, 피사체 없는 안내/텍스트/원단 확대 컷은 자동 제외합니다.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("생성 옵션", expanded=True):
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        target_w = st.number_input("가로(px)", min_value=200, max_value=2000, value=DEFAULT_TARGET_W, step=10)
    with c2:
        target_h = st.number_input("세로(px)", min_value=200, max_value=3000, value=DEFAULT_TARGET_H, step=10)
    with c3:
        preset = st.selectbox("빠른 사이즈", ["기본 450×633", "정사각 1000×1000", "세로 800×1200", "가로 1200×800", "직접 입력"], index=0)
        if preset == "기본 450×633":
            target_w, target_h = 450, 633
        elif preset == "정사각 1000×1000":
            target_w, target_h = 1000, 1000
        elif preset == "세로 800×1200":
            target_w, target_h = 800, 1200
        elif preset == "가로 1200×800":
            target_w, target_h = 1200, 800
    max_images = st.slider("상세영역에서 수집할 최대 이미지 수", 50, 600, 250, step=50)
    skip_no_subject = st.checkbox("피사체 없는 이미지 자동 제외", value=True)
    st.caption("비율 왜곡 없이 Cover 방식으로 채우며, 최종 가장자리 1~3px는 안쪽 픽셀로 덮어 흰줄을 제거합니다.")

all_outputs = []
skipped_all = []

tab1, tab2, tab3 = st.tabs(["① 상세페이지 URL", "② 이미지 주소(URL)", "③ 이미지 업로드"])

with tab1:
    page_url = st.text_input("상세페이지 URL", placeholder="https://.../product/detail.html?product_no=28772")
    if st.button("URL에서 본문 상세이미지 수집 → 썸네일 생성", type="primary", key="go1"):
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
                            outs, skips = process_image_any(pil, f"url{i:03d}", int(target_w), int(target_h), skip_no_subject)
                            all_outputs += outs
                            skipped_all += skips
                        except Exception as e:
                            skipped_all.append((f"url{i:03d}", f"다운로드/처리 실패: {e}"))

with tab2:
    st.write("이미지 URL을 여러 줄로 붙여넣으세요. 각 줄 1개")
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
                        outs, skips = process_image_any(pil, f"img{i:03d}", int(target_w), int(target_h), skip_no_subject)
                        all_outputs += outs
                        skipped_all += skips
                    except Exception as e:
                        skipped_all.append((f"img{i:03d}", f"다운로드/처리 실패: {e}"))

with tab3:
    uploads = st.file_uploader("상세페이지 이미지 업로드 (여러 장 가능)", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True)
    if uploads:
        with st.spinner(f"업로드 이미지 처리 중… ({len(uploads)}개)"):
            for i, f in enumerate(uploads, start=1):
                try:
                    pil = Image.open(f).convert("RGB")
                    base = safe_name(f.name.rsplit(".", 1)[0])
                    outs, skips = process_image_any(pil, f"up{i:03d}_{base}", int(target_w), int(target_h), skip_no_subject)
                    all_outputs += outs
                    skipped_all += skips
                except Exception as e:
                    skipped_all.append((f"up{i:03d}_{f.name}", f"업로드 처리 실패: {e}"))

if all_outputs:
    st.success(f"총 {len(all_outputs)}장 생성 완료 ({int(target_w)}×{int(target_h)})")
    st.subheader("미리보기")
    st.image([img for _, img in all_outputs[:36]], width=180)

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, img in all_outputs:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=95, optimize=True)
            zf.writestr(name, buf.getvalue())
        if skipped_all:
            report = "\n".join([f"{name}\t{reason}" for name, reason in skipped_all])
            zf.writestr("_skipped_images_report.txt", report)
    zip_buf.seek(0)

    st.download_button(
        f"ZIP 다운로드 ({int(target_w)}×{int(target_h)})",
        data=zip_buf,
        file_name=f"misharp_thumbnails_{int(target_w)}x{int(target_h)}.zip",
        mime="application/zip",
    )
else:
    st.info("아직 결과가 없습니다. 위 탭에서 입력 후 생성해보세요.")

if skipped_all:
    with st.expander(f"자동 제외/실패 목록 보기 ({len(skipped_all)}개)", expanded=False):
        for name, reason in skipped_all[:200]:
            st.write(f"- {name}: {reason}")

st.markdown(
    """
    <hr style="margin-top:40px; margin-bottom:10px;">
    <div style="font-size:11px; color:#888; line-height:1.5; text-align:center;">
        ⓒ misharpcompany. All rights reserved.<br>
        본 프로그램의 저작권은 미샵컴퍼니(misharpcompany)에 있으며, 무단 복제·배포·사용을 금합니다.<br>
        본 프로그램은 미샵컴퍼니 내부 직원 전용으로, 외부 유출 및 제3자 제공을 엄격히 금합니다.
        <br><br>
        This program is the intellectual property of misharpcompany.
        Unauthorized copying, distribution, or use is strictly prohibited.
    </div>
    """,
    unsafe_allow_html=True,
)
