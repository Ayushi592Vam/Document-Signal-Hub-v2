"""
ui/topbar.py
Renders the top navigation bar: logo, title, schema badge, settings gear, cache button.
Returns True if the settings dialog should be opened.
"""

import streamlit as st
from modules.logo import logo_img_tag


def _navbar_badge_html(active_schema: str | None, schemas: dict) -> str:
    if not active_schema or active_schema not in schemas:
        return ""
    sc = schemas[active_schema]
    return (
        f'<span style="'
        f'display:inline-flex;align-items:center;gap:6px;'
        f'border-radius:6px;padding:4px 14px;'
        f'font-size:12px;font-weight:700;font-family:monospace;'
        f'border:1px solid {sc["color"]}55;'
        f'color:{sc["color"]};background:{sc["color"]}12;'
        f'white-space:nowrap;letter-spacing:0.3px;margin-left:20px;">'
        f'{sc["icon"]} {active_schema} &nbsp;&middot;&nbsp; {sc["version"]}</span>'
    )


def render_topbar(schemas: dict, config_load_status: dict) -> bool:
    """
    Renders the top bar.
    Returns True if the settings gear was clicked (caller should open dialog).
    """
    active_schema = st.session_state.get("active_schema", None)
    _logo         = logo_img_tag(height=52)
    _badge        = _navbar_badge_html(active_schema, schemas)

    col_title, col_cache, col_gear = st.columns([10, 1, 1])

    with col_title:
        st.markdown(
            '<div style="'
            'display:flex;align-items:center;'
            'padding:10px 0 8px 0;min-height:60px;'
            '">'
            + _logo +
            '<div style="'
            'display:flex;flex-direction:column;'
            'justify-content:center;gap:3px;'
            '">'
            '<span style="'
            'font-size:17px;font-weight:700;color:#ffffff;'
            'font-family:\'Segoe UI\',\'Helvetica Neue\',Arial,sans-serif;'
            'letter-spacing:-0.3px;line-height:1.2;white-space:nowrap;'
            '">&#128737;&nbsp; TPA Loss Run Parser</span>'
            '<span style="'
            'font-size:11px;font-weight:400;color:#8888bb;'
            'font-family:\'JetBrains Mono\',\'Cascadia Code\',\'Consolas\',monospace;'
            'letter-spacing:0.4px;white-space:nowrap;'
            '">Automated Claims Data Ingestion &amp; Multi-Schema Export Platform</span>'
            '</div>'
            + (_badge if _badge else '') +
            '</div>',
            unsafe_allow_html=True,
        )

    with col_cache:
        st.markdown("<div style='padding-top:16px;'>", unsafe_allow_html=True)
        cache_clicked = st.button(
            "🗑", key="open_cache_btn",
            help="Clear app cache",
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_gear:
        st.markdown("<div style='padding-top:16px;'>", unsafe_allow_html=True)
        clicked = st.button(
            "⚙", key="open_settings",
            help="Settings",
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Cache clear confirmation — inline, no dialog ──────────────────────────
    if cache_clicked:
        st.session_state["_show_cache_confirm"] = True

    if st.session_state.get("_show_cache_confirm"):
        st.markdown(
            "<div style='background:#1a1a2e;border:1px solid #2a2a45;"
            "border-left:3px solid #f87171;border-radius:8px;"
            "padding:14px 18px;margin:6px 0 10px 0;'>"
            "<div style='font-size:14px;font-weight:700;color:#f0efff;"
            "margin-bottom:6px;'>Clear Caches</div>"
            "<div style='font-size:13px;color:#a0a0c8;margin-bottom:14px;'>"
            "This will clear parsed sheet data, file duplicate memory, "
            "claim duplicate history, and UI session state.</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        _c1, _c2, _spacer = st.columns([1, 1, 5])
        with _c1:
            if st.button("Cancel", key="cache_cancel_btn", use_container_width=True):
                # Just hide the panel — no rerun so the current UI stays intact
                st.session_state["_show_cache_confirm"] = False
        with _c2:
            if st.button("Clear caches", key="cache_confirm_btn",
                         type="primary", use_container_width=True):
                from modules.cache_manager import (
                    clear_parsed_cache, clear_hash_store, clear_claim_dup_store,
                )
                clear_parsed_cache()
                clear_hash_store()
                clear_claim_dup_store()

                # Reset duplicate flags — file and sheets are now treated as NEW
                st.session_state["is_duplicate_file"]    = False
                st.session_state["duplicate_first_seen"] = None
                st.session_state["duplicate_orig_name"]  = None
                st.session_state["sheet_dup_info"]       = {
                    sn: None for sn in st.session_state.get("sheet_names", [])
                }
                # Also reset the file hash so next upload re-registers as new
                st.session_state["current_file_hash"]    = ""
                st.session_state["last_uploaded"]        = None
                st.session_state["_show_cache_confirm"]  = False

                # Clear all claim/edit/mod state but keep file upload + prefs
                keys_to_keep = {
                    "tmpdir", "sheet_names", "sheet_cache",
                    "selected_idx", "focus_field",
                    "sheet_hashes",
                    "conf_threshold", "use_conf_threshold", "active_schema",
                    "schema_popup_target", "schema_popup_tab",
                    "claim_dup_migrated_v2",
                    "is_duplicate_file", "duplicate_first_seen",
                    "duplicate_orig_name", "sheet_dup_info",
                    "current_file_hash", "last_uploaded",
                }
                for k in [key for key in list(st.session_state.keys())
                          if key not in keys_to_keep
                          and not key.startswith("custom_fields_")]:
                    del st.session_state[k]
                st.toast("✅ Cache cleared — file will appear as new on next upload", icon="🗑")
                st.rerun()

    st.markdown(
        '<hr style="border:none;border-top:1px solid #2a2a45;margin:2px 0 18px 0;">',
        unsafe_allow_html=True,
    )
    return clicked
