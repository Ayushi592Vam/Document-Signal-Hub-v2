"""
modules/pdf_extractor.py

PDF / scanned-image extraction using Azure Document Intelligence.
v3: 
  - JSON export is KV-only (no paragraphs/tables/bboxes in output)
  - Fallback spatial KV mining for two-column / bold-label layouts
    (handles Trellis-style dockets where ADI misses KV pairs)
"""

import io
import os
import re
import tempfile
import datetime
from typing import Any

from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient


# ── Credentials ───────────────────────────────────────────────────────────────
def _get_client() -> DocumentAnalysisClient:
    endpoint = os.environ.get("AZURE_DI_ENDPOINT", "").rstrip("/")
    key      = os.environ.get("AZURE_DI_KEY", "")
    if not endpoint or not key:
        raise EnvironmentError(
            "Azure Document Intelligence credentials not found. "
            "Set AZURE_DI_ENDPOINT and AZURE_DI_KEY in your .env file."
        )
    return DocumentAnalysisClient(endpoint, AzureKeyCredential(key))


# ── Polygon helper ────────────────────────────────────────────────────────────
def _poly(region) -> list[dict] | None:
    if region is None:
        return None
    poly = getattr(region, "polygon", None)
    if not poly:
        return None
    return [{"x": pt.x, "y": pt.y} for pt in poly]


def _bbox_center(bbox: list[dict]) -> tuple[float, float]:
    xs = [p["x"] for p in bbox]
    ys = [p["y"] for p in bbox]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _bbox_top(bbox: list[dict]) -> float:
    return min(p["y"] for p in bbox)


def _bbox_bottom(bbox: list[dict]) -> float:
    return max(p["y"] for p in bbox)


def _bbox_left(bbox: list[dict]) -> float:
    return min(p["x"] for p in bbox)


def _bbox_right(bbox: list[dict]) -> float:
    return max(p["x"] for p in bbox)


# ── Spatial KV fallback for two-column bold-label layouts ─────────────────────
def _spatial_kv_from_paragraphs(paragraphs: list[dict], page_w: float) -> list[dict]:
    """
    For documents like Trellis dockets where ADI doesn't fire KV pairs,
    mine them spatially from paragraphs:

    Strategy A — two-column layout (label left col, value right col):
      Labels have x_right < page_w * 0.55 AND the next paragraph on the
      same or very next line shares the same y-band but x_left > page_w * 0.45

    Strategy B — stacked layout (label on one line, value on the next):
      Label paragraph has role=None, short text (<= 5 words), ALL CAPS or
      title-case, followed immediately by a non-empty value paragraph within
      a small vertical gap.
    """
    results = []
    used    = set()

    # Sort paragraphs by page then top-y
    paras = sorted(paragraphs, key=lambda p: (p["page"], _bbox_top(p["bbox"]) if p.get("bbox") else 0))

    mid_x = page_w * 0.50  # column split heuristic

    # ── Strategy A: side-by-side columns ─────────────────────────────────────
    for i, p in enumerate(paras):
        if i in used or not p.get("bbox"):
            continue
        text = p["text"].strip()
        if not text or len(text.split()) > 6:
            continue
        # Must be in left column
        if _bbox_right(p["bbox"]) > mid_x * 1.1:
            continue
        # Look like a label: title-case or ALL CAPS, short
        if not (_is_label_like(text)):
            continue

        p_top = _bbox_top(p["bbox"])
        p_bot = _bbox_bottom(p["bbox"])
        p_h   = max(p_bot - p_top, 0.01)

        # Find best matching value: same page, right column, y overlaps
        best_val = None
        best_dist = 9999
        for j, q in enumerate(paras):
            if j == i or j in used or not q.get("bbox"):
                continue
            if q["page"] != p["page"]:
                continue
            qval = q["text"].strip()
            if not qval:
                continue
            # Must be in right column
            if _bbox_left(q["bbox"]) < mid_x * 0.9:
                continue
            q_top = _bbox_top(q["bbox"])
            q_bot = _bbox_bottom(q["bbox"])
            # Y overlap or very close vertically (within 1.5× label height)
            v_overlap = min(p_bot, q_bot) - max(p_top, q_top)
            if v_overlap > -p_h * 0.5:
                dist = abs((p_top + p_bot) / 2 - (q_top + q_bot) / 2)
                if dist < best_dist:
                    best_dist = dist
                    best_val  = (j, q)

        if best_val and best_dist < p_h * 2.5:
            j, q = best_val
            results.append({
                "key":        text.rstrip(":").strip(),
                "value":      q["text"].strip(),
                "confidence": 75.0,
                "key_bbox":   p["bbox"],
                "val_bbox":   q["bbox"],
                "page":       p["page"],
                "_source":    "spatial_sidebyside",
            })
            used.add(i)
            used.add(j)

    # ── Strategy B: stacked (label then value on next line) ──────────────────
    for i, p in enumerate(paras):
        if i in used or not p.get("bbox"):
            continue
        text = p["text"].strip()
        if not text or len(text.split()) > 5:
            continue
        if not _is_label_like(text):
            continue

        p_bot = _bbox_bottom(p["bbox"])
        p_h   = max(_bbox_bottom(p["bbox"]) - _bbox_top(p["bbox"]), 0.08)

        # Look for next non-empty para on same page, directly below
        for j in range(i + 1, min(i + 4, len(paras))):
            if j in used or not paras[j].get("bbox"):
                continue
            q = paras[j]
            if q["page"] != p["page"]:
                break
            qval = q["text"].strip()
            if not qval:
                continue
            q_top = _bbox_top(q["bbox"])
            gap   = q_top - p_bot
            if gap < 0 or gap > p_h * 3:
                break
            # Value should NOT itself look like a label (avoid pairing two labels)
            if _is_label_like(qval) and len(qval.split()) <= 3:
                break
            results.append({
                "key":        text.rstrip(":").strip(),
                "value":      qval,
                "confidence": 70.0,
                "key_bbox":   p["bbox"],
                "val_bbox":   q["bbox"],
                "page":       p["page"],
                "_source":    "spatial_stacked",
            })
            used.add(i)
            used.add(j)
            break

    return results


def _is_label_like(text: str) -> bool:
    """True if text looks like a field label (short, title/upper-case words)."""
    text = text.strip().rstrip(":")
    if not text:
        return False
    words = text.split()
    if len(words) > 7:
        return False
    # All caps or title case
    if text == text.upper() and len(text) > 1:
        return True
    if text == text.title():
        return True
    # Mixed case with at least one capitalised word
    cap_words = sum(1 for w in words if w and w[0].isupper())
    return cap_words / len(words) >= 0.6


# ── Main extraction ───────────────────────────────────────────────────────────
def extract_pdf(file_bytes: bytes, filename: str) -> dict[str, Any]:
    tmp_path = None
    result_doc = {
        "filename":     filename,
        "extracted_at": datetime.datetime.now().isoformat(),
        "page_count":   0,
        "pages":        [],
        "kv_pairs":     [],
        "paragraphs":   [],   # kept internally for spatial fallback & bbox tab
        "tables":       [],
        "raw_text":     "",
        "file_bytes":   file_bytes,
        "error":        None,
    }

    try:
        client = _get_client()
        suffix = os.path.splitext(filename)[1] or ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        with open(tmp_path, "rb") as f:
            poller = client.begin_analyze_document("prebuilt-document", f)
        result = poller.result()

        # Page dimensions
        pages_meta = []
        for pg in (result.pages or []):
            pages_meta.append({
                "page":   pg.page_number,
                "width":  pg.width  or 8.5,
                "height": pg.height or 11.0,
                "unit":   pg.unit   or "inch",
            })
        result_doc["page_count"] = len(pages_meta)
        result_doc["pages"]      = pages_meta

        # KV pairs with bounding polygons
        kv_pairs = []
        for kv in (result.key_value_pairs or []):
            if not kv.key:
                continue
            page_num = 1
            key_bbox = val_bbox = None
            if kv.key.bounding_regions:
                br = kv.key.bounding_regions[0]
                page_num = br.page_number
                key_bbox = _poly(br)
            if kv.value and kv.value.bounding_regions:
                vbr = kv.value.bounding_regions[0]
                val_bbox = _poly(vbr)
                if page_num == 1 and vbr.page_number:
                    page_num = vbr.page_number
            kv_pairs.append({
                "key":        kv.key.content.strip() if kv.key else "",
                "value":      kv.value.content.strip() if kv.value else "",
                "confidence": round((kv.confidence or 0) * 100, 1),
                "key_bbox":   key_bbox,
                "val_bbox":   val_bbox,
                "page":       page_num,
                "_source":    "adi",
            })

        # Paragraphs (kept for spatial fallback + bbox overlay)
        paragraphs = []
        for para in (result.paragraphs or []):
            page_num = 1
            bbox = None
            if para.bounding_regions:
                br = para.bounding_regions[0]
                page_num = br.page_number
                bbox = _poly(br)
            paragraphs.append({
                "page": page_num,
                "text": para.content.strip(),
                "role": getattr(para, "role", None),
                "bbox": bbox,
            })
        result_doc["paragraphs"] = paragraphs

        # ── Spatial fallback: fire when ADI found < 3 KV pairs ────────────────
        # (Trellis-style docs, legal dockets, etc. often have 0 ADI KV pairs)
        if len(kv_pairs) < 3 and paragraphs:
            page_w = pages_meta[0]["width"] if pages_meta else 8.5
            spatial_kvs = _spatial_kv_from_paragraphs(paragraphs, page_w)
            # Merge: only add spatial KVs whose key isn't already in ADI results
            adi_keys = {kv["key"].lower() for kv in kv_pairs}
            for skv in spatial_kvs:
                if skv["key"].lower() not in adi_keys:
                    kv_pairs.append(skv)

        result_doc["kv_pairs"] = kv_pairs

        # Tables (kept for internal use / bbox tab only)
        tables = []
        for tbl in (result.tables or []):
            page_num = 1
            if tbl.bounding_regions:
                page_num = tbl.bounding_regions[0].page_number
            grid = [[""] * tbl.column_count for _ in range(tbl.row_count)]
            for cell in tbl.cells:
                grid[cell.row_index][cell.column_index] = cell.content.strip()
            tables.append({"page": page_num, "rows": grid})
        result_doc["tables"] = tables

        # Raw text
        raw_parts = []
        for page in (result.pages or []):
            for line in (page.lines or []):
                raw_parts.append(line.content)
        result_doc["raw_text"] = "\n".join(raw_parts)

    except EnvironmentError as e:
        result_doc["error"] = str(e)
    except Exception as e:
        result_doc["error"] = f"Azure ADI error: {e}"
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    return result_doc


# ── Render page with bounding-box overlays ────────────────────────────────────
def render_page_with_boxes(
    result: dict[str, Any],
    page_number: int,
    highlight_idx: int | None = None,
    dpi: int = 150,
) -> bytes | None:
    try:
        import fitz
        from PIL import Image, ImageDraw
    except ImportError:
        return None

    file_bytes = result.get("file_bytes")
    if not file_bytes:
        return None

    page_meta = next(
        (p for p in result.get("pages", []) if p["page"] == page_number), None
    )
    page_w_in = page_meta["width"]  if page_meta else 8.5
    page_h_in = page_meta["height"] if page_meta else 11.0

    filename = result.get("filename", "")
    suffix   = os.path.splitext(filename)[1].lower()

    if suffix in (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"):
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    else:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pg  = doc[page_number - 1]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = pg.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()

    img_w, img_h = img.size
    scale_x = img_w / page_w_in
    scale_y = img_h / page_h_in

    draw = ImageDraw.Draw(img, "RGBA")

    kv_on_page = [
        (i, kv) for i, kv in enumerate(result.get("kv_pairs", []))
        if kv.get("page") == page_number
    ]

    for i, kv in kv_on_page:
        is_hi    = (i == highlight_idx)
        key_fill = (255, 220,  50, 150) if is_hi else ( 79, 156, 249, 110)
        val_fill = (255, 140,   0, 150) if is_hi else ( 52, 211, 153, 110)
        key_line = (255, 200,   0)      if is_hi else ( 79, 156, 249)
        val_line = (255, 100,   0)      if is_hi else ( 52, 211, 153)
        lw       = 3 if is_hi else 1

        def _box(bbox, fill, outline):
            if not bbox or len(bbox) < 2:
                return
            pts = [(p["x"] * scale_x, p["y"] * scale_y) for p in bbox]
            xs  = [p[0] for p in pts]
            ys  = [p[1] for p in pts]
            draw.rectangle([min(xs), min(ys), max(xs), max(ys)],
                           fill=fill, outline=outline, width=lw)
            if is_hi:
                draw.text((min(xs) + 2, min(ys) - 14),
                          kv["key"][:35], fill=(255, 255, 255))

        _box(kv.get("key_bbox"), key_fill, key_line)
        _box(kv.get("val_bbox"), val_fill, val_line)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── Flatten for export — KV ONLY ──────────────────────────────────────────────
def flatten_for_export(extracted: dict[str, Any]) -> dict[str, Any]:
    """
    Clean export: _meta + key_value_pairs only.
    No paragraphs, no tables, no bboxes.
    """
    kv_flat = {}
    for pair in extracted.get("kv_pairs", []):
        k = pair["key"]
        v = pair["value"]
        if not k:
            continue
        # De-duplicate keys with suffix
        base_k = k
        if k in kv_flat:
            idx = 2
            while f"{k} ({idx})" in kv_flat:
                idx += 1
            k = f"{k} ({idx})"
        kv_flat[k] = v

    return {
        "_meta": {
            "filename":     extracted.get("filename"),
            "extracted_at": extracted.get("extracted_at"),
            "page_count":   extracted.get("page_count"),
            "kv_count":     len(kv_flat),
        },
        "key_value_pairs": kv_flat,
    }