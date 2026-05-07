import io
import re
import zipfile
from urllib.parse import urljoin

import numpy as np
import requests
import streamlit as st
from bs4 import BeautifulSoup
from PIL import Image, ImageFilter

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


@st.cache_data(show_spinner=False, ttl=3600)
def download_image_cached(url: str) -> bytes:
    r = requests.get(url, headers={**HEADERS, "Referer": url}, timeout=20)
    r.raise_for_status()
    return r.content

def download_image(url: str) -> Image.Image:
    return Image.open(io.BytesIO(download_image_cached(url))).convert("RGB")


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


def is_text_or_notice_image(pil_img: Image.Image) -> bool:
    """상품/모델 없이 설명문·사이즈표·혜택·공지 위주인 이미지를 제외합니다.
    원단 확대컷과 행거/디테일컷은 제외하지 않도록 텍스트 패턴이 압도적인 경우만 True.
    """
    img = pil_img.convert("RGB")
    w, h = img.size
    if w < 120 or h < 120:
        return True
    arr = np.array(img).astype(np.int16)
    lum = arr.mean(axis=2)
    sat = arr.max(axis=2) - arr.min(axis=2)
    white_ratio = ((arr[:, :, 0] > 242) & (arr[:, :, 1] > 242) & (arr[:, :, 2] > 242)).mean()
    light_plain = ((lum > 224) & (sat < 24)).mean()
    gray = lum.astype(np.float32)
    edge_density = ((np.abs(np.diff(gray, axis=1)) > 20).mean() + (np.abs(np.diff(gray, axis=0)) > 20).mean()) / 2
    dark_ink = (lum < 115).mean()
    if light_plain > 0.78 and dark_ink > 0.006 and edge_density > 0.010:
        return True
    if white_ratio > 0.86 and edge_density > 0.012:
        return True
    sat_mean = (arr.max(axis=2) - arr.min(axis=2)).mean()
    if sat_mean < 4.0 and light_plain > 0.64 and edge_density > 0.018:
        return True
    return False


def has_usable_subject(pil_img: Image.Image) -> bool:
    """썸네일 소재 판단.
    v7 기준: 모델컷뿐 아니라 행거컷, 원단컷, 허리/밑단/봉제 디테일컷도 허용합니다.
    제외 대상은 피사체 없는 공지/사이즈표/텍스트 카드/빈 배경입니다.
    """
    img = trim_edge_bands(pil_img).convert("RGB")
    w, h = img.size
    if w < 160 or h < 160:
        return False
    if is_text_or_notice_image(img):
        return False
    if looks_like_mobile_or_brand_story(img):
        return False

    arr = np.array(img)
    bbox = subject_bbox(img)
    gray = arr.mean(axis=2).astype(np.float32)
    edge_density = ((np.abs(np.diff(gray, axis=1)) > 18).mean() + (np.abs(np.diff(gray, axis=0)) > 18).mean()) / 2
    sat_mean = (arr.max(axis=2) - arr.min(axis=2)).mean()
    white_ratio = ((arr[:, :, 0] > 242) & (arr[:, :, 1] > 242) & (arr[:, :, 2] > 242)).mean()
    texture_or_product = (edge_density > 0.006 and white_ratio < 0.82) or (sat_mean > 3.0 and white_ratio < 0.88)

    if bbox is None:
        return bool(texture_or_product and min(w, h) >= 180)

    l, t, r, b = bbox
    bw, bh = r - l, b - t
    bbox_ratio = (bw * bh) / float(w * h)
    height_ratio = bh / float(h)
    width_ratio = bw / float(w)

    if height_ratio < 0.18 or width_ratio < 0.08 or bbox_ratio < 0.030:
        return bool(texture_or_product and bbox_ratio > 0.018)

    crop = arr[t:b, l:r]
    if crop.size == 0:
        return False
    crop_white = ((crop[:, :, 0] > 242) & (crop[:, :, 1] > 242) & (crop[:, :, 2] > 242)).mean()
    if crop_white > 0.86 and bbox_ratio < 0.30:
        return False
    return True

# =========================
# Trimming / splitting
# =========================
def trim_edge_bands(pil_img: Image.Image, white_thr: int = 246, black_thr: int = 9):
    """가장자리 흰/검정/연회색 여백을 제거합니다.
    기존 버전보다 상하 흰 여백을 더 강하게 제거해 최종 썸네일 상단/하단 흰줄을 방지합니다.
    """
    img = pil_img.convert("RGB")
    arr = np.array(img).astype(np.int16)
    h, w = arr.shape[:2]
    if h < 20 or w < 20:
        return img

    lum = arr.mean(axis=2)
    sat = arr.max(axis=2) - arr.min(axis=2)

    # 순백/검정 띠 + 연회색 저질감 배경 띠를 같이 제거
    row_white = (arr > white_thr).all(axis=2).mean(axis=1)
    row_black = (arr < black_thr).all(axis=2).mean(axis=1)
    row_light_plain = ((lum > 232) & (sat < 18)).mean(axis=1)
    row_std = arr.std(axis=1).mean(axis=1)
    row_band = ((row_white >= 0.955) | (row_black >= 0.985) | (row_light_plain >= 0.965)) & (row_std <= 18.0)

    col_white = (arr > white_thr).all(axis=2).mean(axis=0)
    col_black = (arr < black_thr).all(axis=2).mean(axis=0)
    col_light_plain = ((lum > 232) & (sat < 18)).mean(axis=0)
    col_std = arr.std(axis=0).mean(axis=1)
    col_band = ((col_white >= 0.955) | (col_black >= 0.985) | (col_light_plain >= 0.965)) & (col_std <= 18.0)

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

    if (right - left + 1) < max(120, w * 0.30) or (bottom - top + 1) < max(120, h * 0.30):
        return img
    return img.crop((left, top, right + 1, bottom + 1))


def remove_text_bands(pil_img: Image.Image):
    """상세페이지 이미지 안에 포함된 설명 텍스트 영역을 잘라냅니다.
    OCR 없이 흰/연회색 배경 + 작은 검정 글자 패턴을 상/하단에서 감지합니다.
    원단/상품 디테일 자체는 유지하고, 글자 블록만 제거하는 보수적 컷입니다.
    """
    img = trim_edge_bands(pil_img).convert("RGB")
    arr = np.array(img).astype(np.int16)
    h, w = arr.shape[:2]
    if h < 160 or w < 120:
        return img

    lum = arr.mean(axis=2)
    sat = arr.max(axis=2) - arr.min(axis=2)
    gray = lum.astype(np.float32)
    edge_row = np.zeros(h, dtype=np.float32)
    if w > 2:
        edge_row = (np.abs(np.diff(gray, axis=1)) > 18).mean(axis=1)
    light_plain = ((lum > 222) & (sat < 24)).mean(axis=1)
    dark_ink = (lum < 120).mean(axis=1)

    # 텍스트 행: 밝고 평평한 바탕에 검정 글자/획이 있는 행
    text_like = (light_plain > 0.72) & ((dark_ink > 0.004) | (edge_row > 0.020))
    plain_light = light_plain > 0.90

    # 상단 텍스트/공지 블록 제거
    top_cut = 0
    scan_top = min(h, max(120, int(h * 0.45)))
    for i in range(scan_top):
        # 텍스트 또는 텍스트 주변 흰 여백이면 계속 넘김
        if text_like[i] or (plain_light[i] and i < scan_top - 1 and text_like[max(0, i-2):min(h, i+8)].any()):
            top_cut = i + 1
        elif top_cut > 0 and i - top_cut > 18:
            break

    # 하단 텍스트/공지 블록 제거
    bottom_cut = h
    scan_bottom = max(0, h - max(120, int(h * 0.45)))
    for i in range(h - 1, scan_bottom - 1, -1):
        if text_like[i] or (plain_light[i] and text_like[max(0, i-8):min(h, i+3)].any()):
            bottom_cut = i
        elif bottom_cut < h and bottom_cut - i > 18:
            break

    # 너무 많이 자르면 원본 유지
    if bottom_cut - top_cut < max(120, int(h * 0.35)):
        return img
    out = img.crop((0, top_cut, w, bottom_cut))
    return trim_edge_bands(out)



def bbox_area(bbox):
    if not bbox:
        return 0
    l,t,r,b=bbox
    return max(0,r-l)*max(0,b-t)


def safe_remove_text_bands(pil_img: Image.Image):
    """상품 보존 우선 텍스트 제거.
    텍스트 제거 결과가 피사체를 반으로 자르거나, 너무 작은 조각을 만들면 원본을 유지합니다.
    """
    original = trim_edge_bands(pil_img).convert("RGB")
    w,h = original.size
    before = subject_bbox(original)
    cleaned = remove_text_bands(original)
    cleaned = trim_edge_bands(cleaned).convert("RGB")
    cw,ch = cleaned.size

    # 실제로 거의 안 잘렸으면 그대로 사용
    if abs(cw-w) < 3 and abs(ch-h) < 3:
        return cleaned

    # 35% 이상 줄어들면 하나의 정상 상품컷을 억지로 자른 가능성이 높음
    if (cw*ch) < (w*h*0.65):
        # 단, 원본 자체가 텍스트/공지 위주면 잘라낸 것을 허용
        if before is not None and has_visual_product(original):
            return original

    after = subject_bbox(cleaned)
    if before is not None and after is not None:
        # 피사체 면적이 크게 손상되면 원본 유지
        if bbox_area(after) < bbox_area(before) * 0.62:
            return original
        # 세로 피사체가 갑자기 절반 이하로 작아지면 원본 유지
        if (after[3]-after[1]) < (before[3]-before[1]) * 0.58:
            return original
    return cleaned


def has_visual_product(pil_img: Image.Image) -> bool:
    """모델/상품/원단/디테일처럼 시각적 피사체가 있는지 보수적으로 판단."""
    img = trim_edge_bands(pil_img).convert("RGB")
    w,h = img.size
    if w < 120 or h < 120:
        return False
    arr=np.array(img).astype(np.int16)
    lum=arr.mean(axis=2)
    sat=arr.max(axis=2)-arr.min(axis=2)
    gray=lum.astype(np.float32)
    edge=((np.abs(np.diff(gray,axis=1))>16).mean()+(np.abs(np.diff(gray,axis=0))>16).mean())/2
    white=((arr[:,:,0]>242)&(arr[:,:,1]>242)&(arr[:,:,2]>242)).mean()
    # 원단 확대컷은 edge/saturation은 낮아도 전체가 흰 배경이 아니고 질감 변화가 있음
    texture_std=float(gray.std())
    bbox=subject_bbox(img)
    if bbox is not None:
        l,t,r,b=bbox
        if (r-l)*(b-t) > w*h*0.035:
            return True
    return (white < 0.82 and (edge > 0.0045 or texture_std > 9.0 or sat.mean() > 2.8))



def image_content_stats(pil_img: Image.Image) -> dict:
    img = trim_edge_bands(pil_img).convert("RGB") if 'trim_edge_bands' in globals() else pil_img.convert("RGB")
    arr = np.array(img).astype(np.int16)
    h, w = arr.shape[:2]
    lum = arr.mean(axis=2)
    sat = arr.max(axis=2) - arr.min(axis=2)
    gray = lum.astype(np.float32)
    edge = ((np.abs(np.diff(gray, axis=1)) > 18).mean() + (np.abs(np.diff(gray, axis=0)) > 18).mean()) / 2 if h > 2 and w > 2 else 0
    white = ((arr[:, :, 0] > 242) & (arr[:, :, 1] > 242) & (arr[:, :, 2] > 242)).mean()
    light_plain = ((lum > 224) & (sat < 24)).mean()
    dark_ink = (lum < 115).mean()
    return {"w": w, "h": h, "edge": float(edge), "white": float(white), "light_plain": float(light_plain), "dark_ink": float(dark_ink), "std": float(gray.std()), "sat_mean": float(sat.mean())}


def looks_like_mobile_or_brand_story(pil_img: Image.Image) -> bool:
    """브랜드스토리, 모바일 캡처 UI, 텍스트 안내 이미지 제외."""
    img = trim_edge_bands(pil_img).convert("RGB")
    w, h = img.size
    arr = np.array(img).astype(np.int16)
    lum = arr.mean(axis=2)
    sat = arr.max(axis=2) - arr.min(axis=2)
    gray = lum.astype(np.float32)
    # 모바일 캡처/브랜드스토리: 밝은 바탕 + 검정 글자/아이콘 + 실제 상품 bbox 없음/작음
    bbox = subject_bbox(img)
    bbox_ratio = 0 if bbox is None else ((bbox[2]-bbox[0])*(bbox[3]-bbox[1]) / float(w*h))
    dark_ink = (lum < 110).mean()
    light_plain = ((lum > 218) & (sat < 30)).mean()
    edge = ((np.abs(np.diff(gray, axis=1)) > 18).mean() + (np.abs(np.diff(gray, axis=0)) > 18).mean()) / 2
    # 좌우/상하 중 한쪽에 텍스트가 몰린 브랜드 설명 컷
    left_dark = (lum[:, :max(1, w//2)] < 115).mean()
    right_dark = (lum[:, w//2:] < 115).mean()
    text_imbalance = max(left_dark, right_dark) > 0.018 and min(left_dark, right_dark) < 0.007
    if light_plain > 0.55 and dark_ink > 0.010 and edge > 0.010 and bbox_ratio < 0.18:
        return True
    if text_imbalance and bbox_ratio < 0.22:
        return True
    # 모바일 화면 캡처: 하단/측면 UI 아이콘, 체크 원형 등 작은 고채도 요소 + 상품 bbox 작음
    bottom = arr[int(h*0.72):, :, :]
    if bottom.size:
        blum = bottom.mean(axis=2)
        bsat = bottom.max(axis=2) - bottom.min(axis=2)
        bottom_ink = (blum < 120).mean()
        bottom_color = (bsat > 60).mean()
        if bbox_ratio < 0.26 and bottom_ink > 0.010 and bottom_color > 0.002 and light_plain > 0.35:
            return True

    # 쇼핑몰 모바일 화면 캡처/캡션 카드: 상품은 작고, 하단 또는 중앙에 글자 박스/아이콘이 함께 있는 경우 제외
    mid = arr[int(h*0.28):int(h*0.72), :, :]
    if mid.size:
        mlum = mid.mean(axis=2)
        msat = mid.max(axis=2) - mid.min(axis=2)
        mid_ink = (mlum < 95).mean()
        mid_white_box = ((mlum > 235) & (msat < 18)).mean()
        if bbox_ratio < 0.24 and (mid_ink > 0.020 or mid_white_box > 0.42) and edge > 0.012:
            return True

    # 넓은 검정/흰색 자막 박스가 있는 스타일컷은 썸네일 소재에서 제외
    row_dark = (lum < 75).mean(axis=1)
    row_white_plain = ((lum > 238) & (sat < 16)).mean(axis=1)
    dark_bar_rows = (row_dark > 0.38).mean()
    white_bar_rows = (row_white_plain > 0.82).mean()
    if bbox_ratio < 0.32 and (dark_bar_rows > 0.035 or white_bar_rows > 0.11) and dark_ink > 0.006:
        return True
    return False


def is_low_value_generated_piece(pil_img: Image.Image) -> bool:
    """빈 배경, 너무 잘린 조각, 의미 없는 UI/텍스트컷 방지."""
    img = trim_edge_bands(pil_img).convert("RGB")
    w, h = img.size
    if w < 160 or h < 160:
        return True
    stt = image_content_stats(img)
    if stt["white"] > 0.90 and stt["edge"] < 0.012:
        return True
    if looks_like_mobile_or_brand_story(img):
        return True
    bbox = subject_bbox(img)
    if bbox is None:
        # 원단 확대컷은 허용, 그러나 거의 단색/빈 배경은 제외
        return not (stt["white"] < 0.82 and (stt["std"] > 9.0 or stt["edge"] > 0.006))
    l, t, r, b = bbox
    bw, bh = r-l, b-t
    bbox_ratio = (bw*bh)/float(w*h)
    # 피사체가 너무 작고 한쪽에 치우친 결과물은 제외
    cx = (l+r)/2.0/w
    cy = (t+b)/2.0/h
    if bbox_ratio < 0.045:
        return True
    if (cx < 0.18 or cx > 0.82 or cy < 0.12 or cy > 0.88) and bbox_ratio < 0.22:
        return True
    # 상품 전체컷인데 피사체가 중앙에서 크게 벗어나면 제외
    if bbox_ratio > 0.13 and (cx < 0.34 or cx > 0.66):
        return True
    # 상품 전체컷이 분할되어 상하가 잘린 조각: 피사체가 위/아래 경계를 동시에 강하게 침범하고 전체가 흰 배경이면 제외
    touches_top = t <= max(3, int(h*0.015))
    touches_bottom = b >= h - max(3, int(h*0.015))
    if touches_top and touches_bottom and stt["white"] > 0.38 and bbox_ratio > 0.28:
        # 원단 클로즈업은 흰 배경 비율이 낮거나 bbox 개념이 약하므로 여기서 제외되지 않음
        return True
    return False


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
    lum = a.mean(axis=2)
    sat = a.max(axis=2) - a.min(axis=2)
    if axis == 0:
        white = (a > white_thr).all(axis=2).mean(axis=1)
        black = (a < black_thr).all(axis=2).mean(axis=1)
        light_plain = ((lum > 236) & (sat < 18)).mean(axis=1)
        std = a.std(axis=1).mean(axis=1)
    else:
        white = (a > white_thr).all(axis=2).mean(axis=0)
        black = (a < black_thr).all(axis=2).mean(axis=0)
        light_plain = ((lum > 236) & (sat < 18)).mean(axis=0)
        std = a.std(axis=0).mean(axis=1)
    flags = ((white > 0.93) | (black > 0.965) | (light_plain > 0.94)) & (std < 18)
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


def _strong_seam_cuts(arr, axis: int, min_piece: int):
    """여백 없이 붙은 사진 경계 탐지. 옷 주름/모델 내부선으로 과분할되지 않도록
    전체 폭/높이의 상당 부분에서 동시에 급변하는 경계만 인정합니다.
    """
    a=arr.astype(np.int16)
    gray=a.mean(axis=2).astype(np.float32)
    if axis==0:
        diff=np.abs(np.diff(gray,axis=0))
        length=a.shape[0]
        cross=a.shape[1]
    else:
        diff=np.abs(np.diff(gray,axis=1))
        length=a.shape[1]
        cross=a.shape[0]
    if length < min_piece*2:
        return []
    mean_score=diff.mean(axis=1 if axis==0 else 0)
    strong_ratio=(diff>28).mean(axis=1 if axis==0 else 0)
    med=float(np.median(mean_score))
    p995=float(np.percentile(mean_score,99.5))
    # 경계선이 화면 대부분을 가로지르는 경우만 인정
    candidates=np.where((mean_score>=max(11.5, med*4.8, p995*0.92)) & (strong_ratio>0.42))[0]+1
    cuts=[]
    for c in candidates:
        if c < min_piece or length-c < min_piece:
            continue
        if cuts and c-cuts[-1] < max(18, min_piece//3):
            prev=cuts[-1]
            if mean_score[c-1] > mean_score[prev-1]:
                cuts[-1]=int(c)
        else:
            cuts.append(int(c))
    return cuts[:3]


def _piece_ok_for_split(piece: Image.Image) -> bool:
    piece=trim_edge_bands(piece)
    w,h=piece.size
    if w<120 or h<120:
        return False
    if is_text_or_notice_image(piece):
        return False
    return has_visual_product(piece)


def split_touching_images(pil_img: Image.Image, target_w: int, target_h: int):
    """상세 이미지 한 장 안에 여러 사진이 위아래/좌우로 붙은 경우만 분리.
    v7 핵심: 텍스트/상품 내부선 때문에 정상 1컷을 억지로 반으로 자르지 않습니다.
    """
    img = trim_edge_bands(pil_img).convert("RGB")
    arr = np.array(img)
    h, w = arr.shape[:2]
    min_h = max(105, int(target_h * 0.20))
    min_w = max(105, int(target_w * 0.30))

    # 상세페이지 캡처/원본은 얇은 흰 구분선(2~6px)으로 사진이 이어지는 경우가 많습니다.
    # 이런 명확한 흰 구분선은 적극 분리하되, 색상 seam 분리는 아래에서 매우 보수적으로 처리합니다.
    solid_h = _solid_gap_cuts(arr, axis=0, min_gap=2)
    solid_v = _solid_gap_cuts(arr, axis=1, min_gap=2)
    seam_h = _strong_seam_cuts(arr, axis=0, min_piece=min_h)
    seam_v = _strong_seam_cuts(arr, axis=1, min_piece=min_w)

    hcuts = sorted(set([c for c in solid_h + seam_h if min_h <= c <= h - min_h]))
    vcuts = sorted(set([c for c in solid_v + seam_v if min_w <= c <= w - min_w]))

    def build_h(cuts):
        bounds=[0]+cuts+[h]
        return [trim_edge_bands(img.crop((0,y1,w,y2))) for y1,y2 in zip(bounds[:-1],bounds[1:]) if y2-y1>=min_h]
    def build_v(cuts):
        bounds=[0]+cuts+[w]
        return [trim_edge_bands(img.crop((x1,0,x2,h))) for x1,x2 in zip(bounds[:-1],bounds[1:]) if x2-x1>=min_w]

    # 명확한 흰/검정 구분선은 상세페이지 이미지 경계로 보고 우선 분리합니다.
    # 분리 후 피사체 없는 조각은 뒤 단계에서 자동 제외합니다.
    if solid_h:
        pieces = build_h(sorted(set([c for c in solid_h if min_h <= c <= h - min_h])))
        if len(pieces) >= 2:
            return pieces
    if solid_v:
        pieces = build_v(sorted(set([c for c in solid_v if min_w <= c <= w - min_w])))
        if len(pieces) >= 2:
            return pieces

    # 여백 없이 붙은 seam 분리는 과분할 방지를 위해 모든 조각이 상품컷일 때만 인정합니다.
    for cuts, builder in [(seam_h, build_h), (seam_v, build_v)]:
        cuts = sorted(set([c for c in cuts if (min_h if builder == build_h else min_w) <= c]))
        if cuts:
            pieces=builder(cuts)
            if len(pieces)>=2 and sum(_piece_ok_for_split(p) for p in pieces) >= 2:
                return pieces
    return [img]


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


def crop_to_single_subject(pil_img: Image.Image, target_w: int = DEFAULT_TARGET_W, target_h: int = DEFAULT_TARGET_H):
    """텍스트/혜택 영역은 버리고, 1개 피사체를 목표 비율에 맞게 최대한 크게 남깁니다.
    상단/하단 흰 여백이 남지 않도록 bbox 주변을 타이트하게 잡고 목표 비율로 재구성합니다.
    """
    img = trim_edge_bands(pil_img).convert("RGB")
    bbox = subject_bbox(img)
    if bbox is None:
        return trim_edge_bands(img)

    w, h = img.size
    l, t, r, b = bbox
    bw, bh = r - l, b - t
    # v7: 디테일컷은 타이트하게, 전체 상품컷은 make_thumbnail preserve 모드에서 별도 처리
    pad_x = max(4, int(bw * 0.035))
    pad_y = max(4, int(bh * 0.025))
    l = clamp(l - pad_x, 0, w - 1)
    r = clamp(r + pad_x, 1, w)
    t = clamp(t - pad_y, 0, h - 1)
    b = clamp(b + pad_y, 1, h)

    bw, bh = r - l, b - t
    target_aspect = float(target_w) / float(target_h)
    box_aspect = bw / max(bh, 1)
    cx = (l + r) / 2.0
    cy = (t + b) / 2.0

    # 목표 비율을 만족하는 최소 crop box를 만든 뒤, 피사체가 중앙에 오도록 이동
    if box_aspect > target_aspect:
        crop_w = bw
        crop_h = crop_w / target_aspect
    else:
        crop_h = bh
        crop_w = crop_h * target_aspect

    # 상품 전체컷은 제작자가 다시 중앙 보정하지 않아도 되도록 여유를 더 둡니다.
    full_subject = (bh / float(h) > 0.48) or (bw * bh / float(w * h) > 0.22)
    margin = 1.12 if full_subject else 1.045
    crop_w *= margin
    crop_h *= margin
    crop_w = min(crop_w, w)
    crop_h = min(crop_h, h)

    left = int(round(cx - crop_w / 2.0))
    top = int(round(cy - crop_h / 2.0))
    left = clamp(left, 0, max(0, w - int(round(crop_w))))
    top = clamp(top, 0, max(0, h - int(round(crop_h))))
    right = clamp(left + int(round(crop_w)), 1, w)
    bottom = clamp(top + int(round(crop_h)), 1, h)

    cropped = img.crop((left, top, right, bottom))
    return trim_edge_bands(cropped)


def _bbox_metrics(pil_img: Image.Image):
    img = trim_edge_bands(pil_img).convert("RGB")
    w, h = img.size
    bbox = subject_bbox(img)
    if bbox is None:
        return None
    l, t, r, b = bbox
    bw, bh = r - l, b - t
    return {
        "bbox": bbox,
        "w": w,
        "h": h,
        "bw": bw,
        "bh": bh,
        "area_ratio": (bw * bh) / float(max(1, w * h)),
        "height_ratio": bh / float(max(1, h)),
        "width_ratio": bw / float(max(1, w)),
        "cx_ratio": ((l + r) / 2.0) / float(max(1, w)),
        "cy_ratio": ((t + b) / 2.0) / float(max(1, h)),
    }


def is_full_product_or_hanger_cut(pil_img: Image.Image) -> bool:
    """행거컷/상품 전체컷 판정.
    이런 컷은 '꽉 채우기'보다 '잘리지 않고 중앙 배치'가 우선입니다.
    """
    m = _bbox_metrics(pil_img)
    if not m:
        return False
    # 세로로 긴 옷/모델/행거컷 또는 화면에서 상품 비중이 큰 전체컷
    return (
        (m["height_ratio"] >= 0.46 and m["width_ratio"] >= 0.28)
        or (m["area_ratio"] >= 0.20 and m["height_ratio"] >= 0.36)
        or (m["height_ratio"] >= 0.58)
    )


def crop_subject_preserve_aspect(pil_img: Image.Image, target_w: int, target_h: int):
    """상품 전체컷용 안전 crop.
    bbox에 충분한 좌우/상하 안전마진을 주고, 목표 비율을 맞추되 피사체가 절대 잘리지 않게 합니다.
    """
    img = trim_edge_bands(pil_img).convert("RGB")
    w, h = img.size
    bbox = subject_bbox(img)
    if bbox is None:
        return img
    l, t, r, b = bbox
    bw, bh = r - l, b - t

    # 전체 상품컷은 소매/어깨가 bbox 밖으로 빠지는 경우가 많아 여백을 넉넉히 둡니다.
    pad_x = max(16, int(bw * 0.16))
    pad_top = max(12, int(bh * 0.10))
    pad_bottom = max(12, int(bh * 0.08))
    l = clamp(l - pad_x, 0, w - 1)
    r = clamp(r + pad_x, 1, w)
    t = clamp(t - pad_top, 0, h - 1)
    b = clamp(b + pad_bottom, 1, h)

    bw, bh = r - l, b - t
    target_aspect = float(target_w) / float(target_h)
    crop_w, crop_h = float(bw), float(bh)
    if crop_w / max(1.0, crop_h) > target_aspect:
        crop_h = crop_w / target_aspect
    else:
        crop_w = crop_h * target_aspect

    # 너무 타이트하지 않게 한 번 더 확장. 단, 원본 범위는 넘지 않음.
    crop_w = min(float(w), crop_w * 1.03)
    crop_h = min(float(h), crop_h * 1.03)

    cx = (l + r) / 2.0
    cy = (t + b) / 2.0
    left = int(round(cx - crop_w / 2.0))
    top = int(round(cy - crop_h / 2.0))
    left = clamp(left, 0, max(0, w - int(round(crop_w))))
    top = clamp(top, 0, max(0, h - int(round(crop_h))))
    right = clamp(left + int(round(crop_w)), 1, w)
    bottom = clamp(top + int(round(crop_h)), 1, h)
    return img.crop((left, top, right, bottom))


def dominant_edge_bg_color(pil_img: Image.Image):
    """캔버스 배경용 색상.
    이미지를 늘리거나 블러 배경을 깔지 않고, 상세페이지 배경과 가까운 단색으로만 채웁니다.
    """
    img = pil_img.convert("RGB")
    arr = np.array(img).astype(np.int16)
    h, w = arr.shape[:2]
    band = max(2, min(h, w) // 18)
    border = np.concatenate([
        arr[:band, :, :].reshape(-1, 3),
        arr[h-band:, :, :].reshape(-1, 3),
        arr[:, :band, :].reshape(-1, 3),
        arr[:, w-band:, :].reshape(-1, 3),
    ], axis=0)
    col = np.median(border, axis=0).astype(int)
    return tuple(int(clamp(int(c), 0, 255)) for c in col)


def pad_to_target_aspect(pil_img: Image.Image, target_w: int, target_h: int):
    """비율 왜곡 없이 목표 비율로 패딩합니다.
    절대 금지: 세로/가로로 억지 늘리기, 블러 배경, 원본 비율 파괴.
    """
    img = pil_img.convert("RGB")
    w, h = img.size
    if w <= 0 or h <= 0:
        return Image.new("RGB", (target_w, target_h), (255, 255, 255))
    target_aspect = target_w / float(target_h)
    cur_aspect = w / float(h)
    if abs(cur_aspect - target_aspect) < 0.006:
        return img
    bg = dominant_edge_bg_color(img)
    if cur_aspect > target_aspect:
        new_w = w
        new_h = int(round(w / target_aspect))
    else:
        new_h = h
        new_w = int(round(h * target_aspect))
    canvas = Image.new("RGB", (new_w, new_h), bg)
    x = (new_w - w) // 2
    y = (new_h - h) // 2
    canvas.paste(img, (x, y))
    return canvas


def resize_no_distort(pil_img: Image.Image, target_w: int, target_h: int):
    """목표 사이즈로 저장하되 원본 비율을 절대 왜곡하지 않습니다.
    먼저 목표 비율로 패딩한 뒤 동일비율 리사이즈합니다.
    """
    img = pad_to_target_aspect(pil_img, target_w, target_h)
    return img.resize((target_w, target_h), Image.LANCZOS)


def resize_contain_centered(pil_img: Image.Image, target_w: int, target_h: int):
    """상품 전체 보존 모드.
    v7의 블러/cover 배경 때문에 위아래가 늘어나 보이던 문제를 제거했습니다.
    """
    return resize_no_distort(pil_img, target_w, target_h)


def is_cut_or_offcenter_thumb(pil_img: Image.Image) -> bool:
    """최종 썸네일 검수: 상품 전체컷이 한쪽으로 치우치거나 잘린 경우 제외/재처리 판단."""
    img = pil_img.convert("RGB")
    w, h = img.size
    bbox = subject_bbox(img)
    if bbox is None:
        return False
    l, t, r, b = bbox
    bw, bh = r - l, b - t
    cx = (l + r) / 2.0 / float(w)
    area_ratio = (bw * bh) / float(w * h)
    # 큰 피사체인데 좌우 끝에 닿거나 중심이 많이 벗어나면 제작자가 다시 손봐야 하는 컷으로 판단
    touches_lr = l <= int(w * 0.015) or r >= w - int(w * 0.015)
    offcenter = cx < 0.43 or cx > 0.57
    if area_ratio > 0.18 and (touches_lr or offcenter):
        return True
    return False


def make_thumbnail(pil_img: Image.Image, target_w: int, target_h: int):
    # 행거컷/상품 전체컷은 잘림 방지 + 중앙 배치 + 비율 무왜곡이 최우선입니다.
    if is_full_product_or_hanger_cut(pil_img):
        cut = crop_subject_preserve_aspect(pil_img, target_w, target_h)
        out = resize_no_distort(cut, target_w, target_h)
        return edge_bleed_fix(out, n=2)

    # 원단/디테일컷은 화면을 채우되, 결과가 왜곡되거나 과하게 잘리지 않도록 bbox 중심 crop만 사용합니다.
    cut = crop_to_single_subject(pil_img, target_w, target_h)
    cut = pad_to_target_aspect(cut, target_w, target_h)
    out = cut.resize((target_w, target_h), Image.LANCZOS)
    return edge_bleed_fix(out, n=2)


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


@st.cache_data(show_spinner=False, ttl=1800)
def extract_detail_image_urls_only(page_url: str, max_images: int = 180) -> list[str]:
    html = requests.get(page_url, headers=HEADERS, timeout=20).text
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




def split_touching_images_fast(pil_img: Image.Image, target_w: int, target_h: int):
    """빠른 모드 분리: 명확한 흰/검정/연회색 구분선만 자릅니다.
    정상 1컷을 억지로 반으로 자르는 문제와 처리 지연을 막습니다.
    """
    img = trim_edge_bands(pil_img).convert("RGB")
    arr = np.array(img)
    h, w = arr.shape[:2]
    min_h = max(120, int(target_h * 0.22))
    min_w = max(120, int(target_w * 0.32))
    solid_h = [c for c in _solid_gap_cuts(arr, axis=0, min_gap=2) if min_h <= c <= h - min_h]
    solid_v = [c for c in _solid_gap_cuts(arr, axis=1, min_gap=2) if min_w <= c <= w - min_w]
    if solid_h:
        bounds=[0]+sorted(set(solid_h))+[h]
        return [trim_edge_bands(img.crop((0,y1,w,y2))) for y1,y2 in zip(bounds[:-1],bounds[1:]) if y2-y1>=min_h]
    if solid_v:
        bounds=[0]+sorted(set(solid_v))+[w]
        return [trim_edge_bands(img.crop((x1,0,x2,h))) for x1,x2 in zip(bounds[:-1],bounds[1:]) if x2-x1>=min_w]
    return [img]


def downscale_for_processing(pil_img: Image.Image, max_side: int = 1400) -> Image.Image:
    """분석용 과대 이미지를 먼저 줄여 속도를 안정화합니다. 최종 출력도 썸네일 목적이라 품질 손실을 줄인 범위입니다."""
    img = pil_img.convert("RGB")
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    return img.resize((max(1, int(w*scale)), max(1, int(h*scale))), Image.LANCZOS)

# =========================
# Processing
# =========================
def process_image_any(pil_img: Image.Image, prefix: str, target_w: int, target_h: int, skip_no_subject: bool = True, precise_split: bool = False, remove_text: bool = False):
    outputs, skipped = [], []
    base_img = downscale_for_processing(pil_img, max_side=1400)
    pieces = split_touching_images(base_img, target_w, target_h) if precise_split else split_touching_images_fast(base_img, target_w, target_h)

    for idx, piece in enumerate(pieces, start=1):
        piece = trim_edge_bands(piece)
        # 기본값은 텍스트 제거 OFF: 정상 상품 1컷을 반으로 자르는 문제 방지
        if remove_text:
            piece = safe_remove_text_bands(piece)
        if skip_no_subject and (not has_usable_subject(piece) or is_low_value_generated_piece(piece)):
            skipped.append((f"{prefix}_{idx:02d}", "피사체 없음/공지·사이즈표·텍스트·빈배경·잘린조각으로 판단"))
            continue
        thumb = make_thumbnail(piece, target_w, target_h)
        if skip_no_subject and is_low_value_generated_piece(thumb):
            skipped.append((f"{prefix}_{idx:02d}", "최종 썸네일 품질 미달: 치우침/잘림/무의미 컷"))
            continue
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
      <div class="misharp-sub">MISHARP THUMBNAIL GENERATOR V9 FAST FIXED</div>
      <div class="misharp-caption">1장=1피사체 / 흰줄·여백 제거 / 피사체 중앙 배치 / 기본 450×633 + 사용자 지정 사이즈</div>
    </div>
    <div class="rule-box">
      절대원칙: 썸네일 1개에는 피사체 1개만 남깁니다. 상품 전체컷/행거컷은 잘림 없이 중앙 정렬을 우선하고, 잘린 조각/브랜드스토리/공지/텍스트/빈 이미지/모바일 UI 캡처는 제외합니다. 원단·행거·디테일컷은 상품 피사체가 명확할 때만 포함합니다.
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
    max_images = st.slider("상세영역에서 수집할 최대 이미지 수", 30, 250, 120, step=30)
    skip_no_subject = st.checkbox("피사체 없는 이미지 자동 제외", value=True)
    precise_split = st.checkbox("정밀 분리 모드(느림): 흰 경계선 없는 붙은 이미지까지 탐지", value=False)
    remove_text = st.checkbox("이미지 안 설명 텍스트 영역 제거 시도(느림/위험): 기본 OFF 권장", value=False)
    st.caption("기본은 빠른 모드입니다. 원본 비율은 절대 왜곡하지 않고, 상품 전체컷은 잘림 방지와 중앙 배치를 우선합니다.")

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
                progress = st.progress(0)
                with st.spinner(f"빠른 모드로 다운로드 및 처리 중… ({len(urls)}개)"):
                    for i, u in enumerate(urls, start=1):
                        try:
                            pil = download_image(u)
                            outs, skips = process_image_any(pil, f"url{i:03d}", int(target_w), int(target_h), skip_no_subject, precise_split, remove_text)
                            all_outputs += outs
                            skipped_all += skips
                        except Exception as e:
                            skipped_all.append((f"url{i:03d}", f"다운로드/처리 실패: {e}"))
                        progress.progress(i / max(1, len(urls)))

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
                        outs, skips = process_image_any(pil, f"img{i:03d}", int(target_w), int(target_h), skip_no_subject, precise_split, remove_text)
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
                    outs, skips = process_image_any(pil, f"up{i:03d}_{base}", int(target_w), int(target_h), skip_no_subject, precise_split, remove_text)
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
