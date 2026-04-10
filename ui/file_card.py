"""
ui/file_card.py
Renders the file-ingestion summary card (format, size, hash, duplicate status).
"""

import os
import streamlit as st


def render_file_card(
    uploaded,
    excel_path: str,
    file_hash: str,
    is_dup: bool,
    sheet_dup_info: dict,
    sheet_names: list,
) -> None:
    file_format  = os.path.splitext(uploaded.name)[1].upper().lstrip(".")
    n_sheets     = len(sheet_names)
    file_size_b  = os.path.getsize(excel_path)
    file_size    = (
        f"{file_size_b / 1024:.1f} KB"
        if file_size_b < 1_000_000
        else f"{file_size_b / 1_048_576:.2f} MB"
    )
    badge_cls   = "badge-duplicate" if is_dup else "badge-unique"
    badge_lbl   = "DUPLICATE" if is_dup else "UNIQUE"
    n_dup_sheets = sum(1 for v in sheet_dup_info.values() if v is not None)
    dup_note = (
        "<span style='font-size:11px;color:#f5c842;font-family:var(--mono);'>⚠ Already processed</span>"
        if is_dup
        else "<span style='font-size:11px;color:#34d399;font-family:var(--mono);'>✓ New file ingested</span>"
    )

    def _sheet_pill(sn: str) -> str:
        info = sheet_dup_info.get(sn)
        if info:
            title = (
                f"Sheet already seen in {info['filename']} "
                f"(sheet: {info['sheet_name']}) on {info['first_seen'][:10]}"
            )
            return (
                f"<span class='sheet-pill-sm' title='{title}' "
                f"style='border-color:#f5c842;color:#f5c842;'>⚠ {sn}</span>"
            )
        return f"<span class='sheet-pill-sm'>{sn}</span>"

    sheet_pills        = "".join(_sheet_pill(sn) for sn in sheet_names)
    _file_status_color = (
        "#f5c842" if is_dup else ("#f5c842" if n_dup_sheets > 0 else "#34d399")
    )
    _file_status_text  = (
        "Duplicate" if is_dup else (f"{n_dup_sheets} sheet(s) duplicate" if n_dup_sheets > 0 else "New")
    )

    st.markdown(
        f"""
        <div class="file-card">
          <div class="file-card-header">
            <div class="file-card-title">📄 {uploaded.name}
              <span class="file-badge {badge_cls}">{badge_lbl}</span>
            </div>
            {dup_note}
          </div>
          <div class="file-card-body">
            <div class="file-stat">
              <div class="file-stat-lbl">Format</div>
              <div class="file-stat-val accent">{file_format}</div>
            </div>
            <div class="file-stat">
              <div class="file-stat-lbl">Sheets</div>
              <div class="file-stat-val accent">{n_sheets}</div>
            </div>
            <div class="file-stat">
              <div class="file-stat-lbl">File Size</div>
              <div class="file-stat-val">{file_size}</div>
            </div>
            <div class="file-stat">
              <div class="file-stat-lbl">Status</div>
              <div class="file-stat-val" style="color:{_file_status_color};">{_file_status_text}</div>
            </div>
          </div>
          <div class="file-card-sheets">
            <div class="section-lbl">Worksheets</div>
            {sheet_pills}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Duplicate warnings
    if is_dup:
        first_seen  = st.session_state.get("duplicate_first_seen", "unknown")
        orig_name   = st.session_state.get("duplicate_orig_name", uploaded.name)
        st.warning(
            f"⚠ **Duplicate file detected.** First processed on **{first_seen}** "
            f"(original: `{orig_name}`)."
        )
    elif n_dup_sheets > 0:
        _dup_names   = [sn for sn, v in sheet_dup_info.items() if v is not None]
        _dup_details = "; ".join(
            f"**{sn}** → seen in `{sheet_dup_info[sn]['filename']}` on "
            f"{sheet_dup_info[sn]['first_seen'][:10]}"
            for sn in _dup_names
        )
        st.warning(
            f"⚠ **{n_dup_sheets} sheet(s) already processed in a different file.** {_dup_details}",
            icon="📋",
        )
