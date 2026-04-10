"""
modules/logo.py
Loads the ValueMomentum logo — no cropping, no padding, no background box.
Supports .png, .jpg, .jpeg
"""

import base64
import os


def _load_logo_b64() -> tuple[str, str]:
    """Returns (base64_string, mime_type) or ('', '')."""
    candidates = [
        "valuemomentum_logo.png",
        "valuemomentum_logo.jpg",
        "valuemomentum_logo.jpeg",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "valuemomentum_logo.png"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "valuemomentum_logo.jpg"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "valuemomentum_logo.jpeg"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "valuemomentum_logo.png"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "valuemomentum_logo.jpg"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "valuemomentum_logo.jpeg"),
    ]
    for path in candidates:
        if os.path.exists(path):
            ext  = os.path.splitext(path)[1].lower()
            mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode(), mime
    return "", ""


LOGO_B64, LOGO_MIME = _load_logo_b64()


def logo_img_tag(height: int = 52) -> str:
    """
    Returns a clean <img> tag — no background, no padding, no border-radius,
    no clipping. The logo renders exactly as the source image.
    """
    if not LOGO_B64:
        return ""
    mime = LOGO_MIME or "image/png"
    return (
        f'<img src="data:{mime};base64,{LOGO_B64}" '
        f'style="height:{height}px;width:auto;'
        f'display:inline-block;vertical-align:middle;'
        f'margin-right:18px;flex-shrink:0;" />'
    )
