import io
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin

import numpy as np
import requests
import streamlit as st
from bs4 import BeautifulSoup
from PIL import Image

DEFAULT_TARGET_W, DEFAULT_TARGET_H = 450, 633

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
    )
}

def safe_name(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^\w\-.가-힣]+", "_", s)
    return s[:120] if s else "item"

def download_image(url: str) -> Image.Image:
    r = requests.get(url, headers={**HEADERS, "Referer": url}, timeout=20)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def estimate_background_color(arr_rgb: np.ndarray) -> np.ndarray:
    h, w = arr_rgb.shape[:2]
    band = max(2, min(h, w) // 28)
    border = np.concatenate([
        arr_rgb[:band,:,:].reshape(-1,3),
        arr_rgb[h-band:,:,:].reshape(-1,3),
        arr_rgb[:,:band,:].reshape(-1,3),
        arr_rgb[:,w-band:,:].reshape(-1,3),
    ], axis=0).astype(np.int16)
    return np.median(border, axis=0)

def subject_mask(arr: np.ndarray) -> np.ndarray:
    arr16 = arr.astype(np.int16)
    bg = estimate_background_color(arr16)
    diff = np.sqrt(((arr16 - bg) ** 2).sum(axis=2))
    lum = arr16.mean(axis=2)
    bg_lum = float(bg.mean())
    sat = arr16.max(axis=2) - arr16.min(axis=2)
    mask = (diff > 22) | (np.abs(lum - bg_lum) > 16) | ((sat > 28) & (diff > 12))
    white_bg = (arr16[:,:,0] > 245) & (arr16[:,:,1] > 245) & (arr16[:,:,2] > 245)
    return mask & (~white_bg)

def component_bboxes(mask: np.ndarray, min_area_ratio: float = 0.0012):
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
            comps.append((int(x), int(y), int(x+bw), int(y+bh), int(area)))
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
        left, top, right, bottom, _ = valid[0]
    else:
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
    white_ratio = ((arr[:,:,0] > 242) & (arr[:,:,1] > 242) & (arr[:,:,2] > 242)).mean()
    gray = arr.mean(axis=2).astype(np.float32)
    edge_density = ((np.abs(np.diff(gray, axis=1)) > 22).mean() + (np.abs(np.diff(gray, axis=0)) > 22).mean()) / 2
    sat_mean = (arr.max(axis=2) - arr.min(axis=2)).mean()
    if sat_mean < 1.3 and edge_density > 0.007:
        return False
    if height_ratio < 0.24 or width_ratio < 0.10 or bbox_ratio < 0.045:
        return False
    if white_ratio > 0.70 and bbox_ratio < 0.22 and edge_density > 0.022:
        return False
    if white_ratio > 0.82 and height_ratio < 0.62:
        return False
    crop = arr[t:b, l:r]
    if crop.size == 0:
        return False
    crop_white = ((crop[:,:,0] > 242) & (crop[:,:,1] > 242) & (crop[:,:,2] > 242)).mean()
    if crop_white > 0.78 and bbox_ratio < 0.35:
        return False
    if bbox_ratio > 0.90 and edge_density > 0.035:
        return False
    return True

# =========================
# Trimming
# =========================
def trim_edge_bands(pil_img: Image.Image, white_thr: int = 246, black_thr: int = 9):
    """
    가장자리 흰/검정 여백 제거.
    [V5 수정] solid_ratio_thr 0.985 → 0.75 완화:
      순백(255)이 아닌 연한 배경(~230~246)도 여백으로 인식.
      url006/007/008 타입의 상단 91% 그레이 여백 제거 가능.
    """
    img = pil_img.convert("RGB")
    arr = np.array(img).astype(np.int16)
    h, w = arr.shape[:2]
    # 완화된 기준: 행의 75% 이상이 흰색 계열(>230)이고 std가 낮으면 여백 행으로 판단
    solid_ratio_thr = 0.75
    std_thr = 18.0

    row_white = (arr > white_thr).all(axis=2).mean(axis=1)
    row_lt    = (arr > 230).all(axis=2).mean(axis=1)   # 연한 배경(>230) 포함
    row_black = (arr < black_thr).all(axis=2).mean(axis=1)
    row_std   = arr.std(axis=1).mean(axis=1)
    row_band  = (
        ((row_white >= 0.985) | (row_black >= 0.985))   # 기존: 순백/순흑
        | ((row_lt >= solid_ratio_thr) & (row_std <= std_thr))  # 신규: 연배경 여백
    )

    col_white = (arr > white_thr).all(axis=2).mean(axis=0)
    col_lt    = (arr > 230).all(axis=2).mean(axis=0)
    col_black = (arr < black_thr).all(axis=2).mean(axis=0)
    col_std   = arr.std(axis=0).mean(axis=1)
    col_band  = (
        ((col_white >= 0.985) | (col_black >= 0.985))
        | ((col_lt >= solid_ratio_thr) & (col_std <= std_thr))
    )

    top = 0
    while top < h - 1 and row_band[top]: top += 1
    bottom = h - 1
    while bottom > top and row_band[bottom]: bottom -= 1
    left = 0
    while left < w - 1 and col_band[left]: left += 1
    right = w - 1
    while right > left and col_band[right]: right -= 1

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
        score = np.abs(np.diff(gray, axis=0)).mean(axis=1)
        length = arr.shape[0]
    else:
        score = np.abs(np.diff(gray, axis=1)).mean(axis=0)
        length = arr.shape[1]
    if len(score) < min_piece * 2:
        return []
    med = float(np.median(score))
    p98 = float(np.percentile(score, 98))
    thr = max(med * 4.0, p98 * 1.18, 10.0)
    candidates = np.where(score >= thr)[0] + 1
    cuts = []
    for c in candidates:
        if c < min_piece or length - c < min_piece:
            continue
        if cuts and c - cuts[-1] < min_piece:
            prev = cuts[-1]
            if score[c - 1] > score[prev - 1]:
                cuts[-1] = int(c)
        else:
            cuts.append(int(c))
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

    return pieces if pieces else [img]

# =========================
# Thumbnail generation  (V6 완전 재설계)
# =========================

def edge_bleed_fix(pil_img: Image.Image, n: int = 3):
    """가장자리 1~3px를 안쪽 픽셀로 덮어 흰줄 제거."""
    img = pil_img.convert("RGB")
    arr = np.array(img).copy()
    h, w = arr.shape[:2]
    n = max(1, min(n, 5))
    if h <= 2 * n + 2 or w <= 2 * n + 2:
        return img
    arr[0:n,:,:]      = arr[n:n+1,:,:]
    arr[h-n:h,:,:]    = arr[h-n-1:h-n,:,:]
    arr[:,0:n,:]      = arr[:,n:n+1,:]
    arr[:,w-n:w,:]    = arr[:,w-n-1:w-n,:]
    return Image.fromarray(arr)


def make_thumbnail(pil_img: Image.Image, target_w: int, target_h: int):
    """
    [V6 핵심 재설계] 피사체 bbox 기반 안전 크롭.

    기존 문제:
      crop_to_single_subject → 피사체만 남김 → 이미지 비율이 target과 유사해짐
      → resize_cover가 세로를 잘라버림 → 상단/하단 잘림 발생

    새 전략:
      1. trim_edge_bands로 여백만 제거 (피사체 crop 없음)
      2. subject_bbox로 피사체 위치 파악
      3. 항상 가로를 target_w에 맞게 scale (좌우 여백 0 보장)
      4. scale 후 피사체 bbox가 target_h 안에 들어오는지 확인
         - 들어옴 → 피사체 상단 기준 crop (머리 우선)
         - 안 들어옴(전신샷) → 피사체 상단 28% 기준 crop (얼굴 최대 확보)
      5. 피사체 상단이 절대 잘리지 않도록 crop_top clamp
    """
    img = trim_edge_bands(pil_img).convert("RGB")
    W, H = img.size

    # 피사체 bbox 파악
    bbox = subject_bbox(img)

    if bbox is None:
        # fallback: bbox 없으면 가로 기준 scale 후 상단 기준 crop
        scale = target_w / W
        new_w = int(round(W * scale))
        new_h = int(round(H * scale))
        resized = img.resize((new_w, new_h), Image.LANCZOS)
        crop_top = 0
        if new_h > target_h:
            crop_top = max(0, new_h // 2 - target_h // 2)
        crop_top = min(crop_top, max(0, new_h - target_h))
        out = resized.crop((0, crop_top, target_w, min(new_h, crop_top + target_h)))
        if out.size[1] < target_h:
            out = out.resize((target_w, target_h), Image.LANCZOS)
        return edge_bleed_fix(out)

    bl, bt, br, bb = bbox
    subj_w = br - bl
    subj_h = bb - bt

    # ── Step 1. 가로를 target_w에 맞게 scale ──────────────────────────
    scale = target_w / W
    new_w = int(round(W * scale))
    new_h = int(round(H * scale))

    # scale된 피사체 좌표
    s_bl = bl * scale
    s_bt = bt * scale
    s_br = br * scale
    s_bb = bb * scale
    s_subj_h = subj_h * scale

    # ── Step 2. crop_top 결정 ──────────────────────────────────────────
    if s_subj_h <= target_h:
        # 피사체가 target_h 안에 들어옴
        # 피사체 상단 위로 여백을 두고 싶지만, 피사체가 잘리면 안 됨
        # → 피사체 상단을 target 상단 10% 위치에 맞춤 (머리 위 약간 여백)
        crop_top = s_bt - target_h * 0.08
        crop_top = max(0.0, crop_top)
        # 피사체 하단이 잘리지 않도록
        if crop_top + target_h < s_bb:
            crop_top = s_bb - target_h
        crop_top = max(0.0, crop_top)
    else:
        # 피사체가 target_h보다 긺 → 얼굴/상단 최대 확보
        # 피사체 상단에서 아래로 28% 지점을 화면 상단 20% 위치에
        face_y   = s_bt + s_subj_h * 0.28
        crop_top = face_y - target_h * 0.20
        crop_top = max(0.0, crop_top)
        # 피사체 상단이 화면 밖으로 나가지 않게 (머리 잘림 방지)
        if crop_top > s_bt:
            crop_top = max(0.0, s_bt - 5)

    crop_top = int(round(min(crop_top, max(0, new_h - target_h))))

    # ── Step 3. 리사이즈 후 크롭 ──────────────────────────────────────
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    crop_bot = crop_top + target_h

    if crop_bot > new_h:
        # 이미지가 target_h보다 짧으면 전체를 target 크기로 늘림
        crop_region = resized.crop((0, crop_top, target_w, new_h))
        out = crop_region.resize((target_w, target_h), Image.LANCZOS)
    else:
        out = resized.crop((0, crop_top, target_w, crop_bot))

    return edge_bleed_fix(out)

# =========================
# URL extraction
# =========================
DETAIL_CONTAINER_SELECTORS = [
    "#prdDetailContent","#prdDetail",".xans-product-detail",
    ".xans-product-detaildesign",".xans-product-additional",
    "#productDetail",".cont_detail",".detailArea",
]

def extract_detail_image_urls_only(page_url: str, max_images: int = 250) -> list:
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
def process_image_any(pil_img, prefix, target_w, target_h, skip_no_subject=True):
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

def _download_and_process(args):
    i, url, prefix, target_w, target_h, skip_no_subject = args
    try:
        pil = download_image(url)
        outs, skips = process_image_any(pil, prefix, target_w, target_h, skip_no_subject)
        return i, outs, skips
    except Exception as e:
        return i, [], [(prefix, f"실패: {e}")]

def run_with_progress(urls, prefix_fmt, target_w, target_h, skip_no_subject, max_workers=4, progress_container=None):
    total = len(urls)
    all_outputs, skipped_all = [], []
    args_list = [(i, url, prefix_fmt.format(i), target_w, target_h, skip_no_subject)
                 for i, url in enumerate(urls, start=1)]
    completed = 0
    if progress_container:
        pb = progress_container.progress(0, text=f"⏳ 처리 중... 0 / {total}")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_download_and_process, a): a for a in args_list}
        for future in as_completed(futures):
            i, outs, skips = future.result()
            all_outputs += outs
            skipped_all += skips
            completed += 1
            if progress_container:
                pct = completed / total
                pb.progress(pct, text=f"⏳ 처리 중... {completed} / {total}  ({int(pct*100)}%)")
    if progress_container:
        progress_container.progress(1.0, text=f"✅ 완료! 총 {total}개 처리됨")
    return all_outputs, skipped_all

# =========================
# Streamlit UI
# =========================
st.set_page_config(
    page_title="미샵 썸네일 생성기 | MISHARP Thumbnail Generator",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# SEO 메타태그 + 구조화 데이터 주입
st.markdown("""
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="description" content="미샵 상세페이지 이미지에서 쇼핑몰 등록용 썸네일을 자동 생성하는 도구. 피사체 자동 감지, 여백 제거, 450×633 리사이즈, ZIP 일괄 다운로드 지원. misharpcompany 내부 전용.">
  <meta name="keywords" content="미샵, 썸네일 생성기, 쇼핑몰 썸네일, 상품 이미지, Cafe24, 이미지 리사이즈, 상세페이지, MISHARP, misharpcompany">
  <meta name="author" content="misharpcompany">
  <meta name="robots" content="noindex, nofollow">
  <meta property="og:title" content="미샵 썸네일 생성기 | MISHARP Thumbnail Generator">
  <meta property="og:description" content="상세페이지 URL 입력 한 번으로 쇼핑몰 썸네일 자동 생성. 피사체 중앙 배치, 흰여백 자동 제거, 450×633 일괄 출력.">
  <meta property="og:type" content="website">
  <meta property="og:site_name" content="MISHARP Tools">
  <script type="application/ld+json">
  {
    "@context": "https://schema.org",
    "@type": "WebApplication",
    "name": "미샵 썸네일 생성기",
    "alternateName": "MISHARP Thumbnail Generator",
    "description": "쇼핑몰 상세페이지 이미지에서 썸네일을 자동 추출·생성하는 내부 도구",
    "applicationCategory": "UtilitiesApplication",
    "author": {
      "@type": "Organization",
      "name": "misharpcompany",
      "url": "https://misharp.co.kr"
    },
    "offers": {"@type": "Offer", "price": "0"}
  }
  </script>
</head>
<style>
  .misharp-title-wrap{margin-top:8px;margin-bottom:6px;}
  .misharp-title{font-size:1.55rem;font-weight:800;letter-spacing:-0.02em;margin:0;}
  .misharp-sub{font-size:0.78rem;color:#888;margin-top:4px;}
</style>
<div class="misharp-title-wrap">
  <div class="misharp-title">MISHARP 썸네일 생성기</div>
  <div class="misharp-sub">MISHARP Thumbnail Generator V6 · misharpcompany</div>
</div>
""", unsafe_allow_html=True)

with st.expander("생성 옵션", expanded=True):
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        target_w = st.number_input("가로(px)", min_value=200, max_value=2000, value=DEFAULT_TARGET_W, step=10)
    with c2:
        target_h = st.number_input("세로(px)", min_value=200, max_value=3000, value=DEFAULT_TARGET_H, step=10)
    with c3:
        preset = st.selectbox("빠른 사이즈", ["기본 450×633","정사각 1000×1000","세로 800×1200","가로 1200×800","직접 입력"], index=0)
        if preset == "기본 450×633": target_w, target_h = 450, 633
        elif preset == "정사각 1000×1000": target_w, target_h = 1000, 1000
        elif preset == "세로 800×1200": target_w, target_h = 800, 1200
        elif preset == "가로 1200×800": target_w, target_h = 1200, 800
    col_a, col_b = st.columns(2)
    with col_a:
        max_images = st.slider("최대 이미지 수", 50, 600, 250, step=50)
        skip_no_subject = st.checkbox("피사체 없는 이미지 자동 제외", value=True)
    with col_b:
        max_workers = st.slider("동시 처리 수 (빠를수록 서버 부하↑)", 1, 8, 4, step=1)

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
                st.error("본문(상세영역)에서 이미지 URL을 찾지 못했습니다.")
            else:
                st.info(f"📦 이미지 {len(urls)}개 수집됨. 썸네일 생성 중...")
                prog = st.empty()
                all_outputs, skipped_all = run_with_progress(
                    urls, "url{:03d}", int(target_w), int(target_h),
                    skip_no_subject, max_workers=max_workers, progress_container=prog)

with tab2:
    url_text = st.text_area("이미지 주소 목록", height=180, placeholder="https://.../a.jpg\nhttps://.../b.jpg\n...")
    if st.button("이미지 주소로 생성", type="primary", key="go2"):
        lines = [l.strip() for l in (url_text or "").splitlines() if l.strip()]
        if not lines:
            st.error("이미지 URL을 넣어주세요.")
        else:
            st.info(f"📦 이미지 {len(lines)}개 처리 중...")
            prog2 = st.empty()
            all_outputs, skipped_all = run_with_progress(
                lines, "img{:03d}", int(target_w), int(target_h),
                skip_no_subject, max_workers=max_workers, progress_container=prog2)

with tab3:
    uploads = st.file_uploader("상세페이지 이미지 업로드 (여러 장 가능)", type=["jpg","jpeg","png","webp"], accept_multiple_files=True)
    if uploads:
        st.info(f"📦 업로드 이미지 {len(uploads)}개 처리 중...")
        pa3 = st.empty()
        pb3 = pa3.progress(0, text=f"⏳ 처리 중... 0 / {len(uploads)}")
        for i, f in enumerate(uploads, start=1):
            try:
                pil = Image.open(f).convert("RGB")
                base = safe_name(f.name.rsplit(".", 1)[0])
                outs, skips = process_image_any(pil, f"up{i:03d}_{base}", int(target_w), int(target_h), skip_no_subject)
                all_outputs += outs
                skipped_all += skips
            except Exception as e:
                skipped_all.append((f"up{i:03d}_{f.name}", f"업로드 처리 실패: {e}"))
            pct = i / len(uploads)
            pb3.progress(pct, text=f"⏳ 처리 중... {i} / {len(uploads)}  ({int(pct*100)}%)")
        pa3.progress(1.0, text=f"✅ 완료! 총 {len(uploads)}개 처리됨")

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

st.markdown("""
<hr style="margin-top:40px;margin-bottom:10px;">
<div style="font-size:11px;color:#aaa;line-height:1.6;text-align:center;">
    ⓒ misharpcompany. All rights reserved. 본 프로그램은 미샵컴퍼니 내부 전용입니다.
</div>
""", unsafe_allow_html=True)
