"""Load and expose column configuration from YAML (with fallbacks)."""

from __future__ import annotations

from pathlib import Path

import yaml

from .config import DISPLAY_ORDER_DETAIL, DISPLAY_ORDER_TICKET_LIST, ISSUE_CORE_COLUMNS

_CACHE: dict[str, list[str]] | None = None


def load_column_sets(base_path: str | Path | None = None):
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    base = Path(base_path or Path(__file__).resolve().parent.parent)
    yaml_path = base / "columns.yaml"
    if not yaml_path.exists():
        _CACHE = {
            "detail": list(DISPLAY_ORDER_DETAIL),
            "core": list(ISSUE_CORE_COLUMNS),
            "ticket_list": list(DISPLAY_ORDER_TICKET_LIST),
        }
        return _CACHE
    try:
        data = yaml.safe_load(yaml_path.read_text()) or {}
        sets = data.get("sets", {})
        detail = sets.get("detail") or list(DISPLAY_ORDER_DETAIL)
        core = sets.get("core") or list(ISSUE_CORE_COLUMNS)
        ticket_list = sets.get("ticket_list") or list(DISPLAY_ORDER_TICKET_LIST)
        _CACHE = {"detail": detail, "core": core, "ticket_list": ticket_list}
        return _CACHE
    except Exception:
        _CACHE = {
            "detail": list(DISPLAY_ORDER_DETAIL),
            "core": list(ISSUE_CORE_COLUMNS),
            "ticket_list": list(DISPLAY_ORDER_TICKET_LIST),
        }
        return _CACHE


def get_columns(set_name: str) -> list[str]:
    sets = load_column_sets()
    return sets.get(set_name, [])
