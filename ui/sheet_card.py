"""
ui/sheet_card.py
Renders the sheet-stats card and the LLM field-map notification banner.
"""

import streamlit as st


def render_sheet_card(
    selected_sheet: str,
    sheet_type: str,
    sh_hash: str,
    n_claims: int,
    total_rows: int,
    total_cols: int,
    n_merged: int,
    totals_data: dict,
    n_title_fields: int,
    from_cache: bool,
    sheet_dup_info: dict,
) -> None:
    totals_cls    = "hi" if totals_data else "mid"
    totals_found  = "Found" if totals_data else "None"
    type_cls      = "unk" if sheet_type == "UNKNOWN" else ""

    # Build sub-pieces as plain strings FIRST — avoids any quote nesting inside f-string
    cache_badge = (
        "<span style='font-size:9px;color:#34d399;font-family:monospace;"
        "margin-left:6px;'>&#9889; from cache</span>"
        if from_cache else ""
    )

    html = (
        "<div class='sheet-card'>"
          "<div class='sheet-card-hdr'>"
            "<div class='sheet-card-name'>"
              "⊞ " + selected_sheet + " "
              "<span class='sheet-type-tag " + type_cls + "'>" + sheet_type + "</span>"
              + cache_badge +
            "</div>"
          "</div>"
          "<div class='sheet-stats-grid'>"
            "<div class='sh-stat'>"
              "<div class='sh-stat-lbl'>Claim Rows</div>"
              "<div class='sh-stat-val hi'>" + str(n_claims) + "</div>"
            "</div>"
            "<div class='sh-stat'>"
              "<div class='sh-stat-lbl'>Rows</div>"
              "<div class='sh-stat-val'>" + str(total_rows) + "</div>"
            "</div>"
            "<div class='sh-stat'>"
              "<div class='sh-stat-lbl'>Columns</div>"
              "<div class='sh-stat-val'>" + str(total_cols) + "</div>"
            "</div>"
            "<div class='sh-stat'>"
              "<div class='sh-stat-lbl'>Merged Regions</div>"
              "<div class='sh-stat-val'>" + str(n_merged) + "</div>"
            "</div>"
            "<div class='sh-stat'>"
              "<div class='sh-stat-lbl'>Totals Row</div>"
              "<div class='sh-stat-val " + totals_cls + "'>" + totals_found + "</div>"
            "</div>"
            "<div class='sh-stat'>"
              "<div class='sh-stat-lbl'>Title Fields</div>"
              "<div class='sh-stat-val'>" + str(n_title_fields) + "</div>"
            "</div>"
          "</div>"
        "</div>"
    )

    st.markdown(html, unsafe_allow_html=True)

    # Per-sheet duplicate warning
    _selected_dup = sheet_dup_info.get(selected_sheet)
    if _selected_dup:
        _orig_file  = _selected_dup.get("filename", "unknown file")
        _orig_sheet = _selected_dup.get("sheet_name", selected_sheet)
        _orig_date  = _selected_dup.get("first_seen", "")[:10]
        _sheet_ref  = (
            f"sheet **{_orig_sheet}**"
            if _orig_sheet != selected_sheet
            else "the same sheet name"
        )
        st.warning(
            f"⚠ **This sheet was already processed** — {_sheet_ref} "
            f"in `{_orig_file}` on **{_orig_date}**."
        )


def render_llm_map_banner(llm_map_result: dict, llm_map_count: int) -> None:
    _unmapped_cols = llm_map_result.get("_unmapped", [])
    _details_str   = ", ".join(
        "<b>" + s + "</b> &rarr; " + t
        for s, t in list(llm_map_result.get("mappings", {}).items())[:5]
    )
    _unmapped_str = (
        "<span style='color:var(--red);'>&nbsp;&middot;&nbsp;"
        + str(len(_unmapped_cols)) + " column(s) could not be mapped</span>"
        if _unmapped_cols else ""
    )
    st.markdown(
        "<div class='llm-map-banner'>"
        "<div style='font-size:var(--sz-xs);font-weight:700;color:var(--yellow);"
        "font-family:var(--mono);text-transform:uppercase;letter-spacing:1px;"
        "margin-bottom:4px;'>Unfamiliar columns detected &mdash; "
        + str(llm_map_count) + " automatically mapped</div>"
        "<div style='font-size:var(--sz-xs);color:var(--t2);font-family:var(--font);'>"
        + _details_str + _unmapped_str + "</div>"
        "</div>",
        unsafe_allow_html=True,
    )
