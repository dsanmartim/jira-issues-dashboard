"""Word cloud generation utility (cached)."""

from __future__ import annotations

from io import BytesIO

import streamlit as st

try:
    from wordcloud import WordCloud  # type: ignore
except Exception:  # pragma: no cover
    WordCloud = None  # type: ignore

WORDCLOUD_AVAILABLE = WordCloud is not None


@st.cache_data(show_spinner=False)
def wordcloud_png(text: str, width: int = 600, height: int = 260, bg: str = "white") -> bytes | None:
    if not text or not text.strip() or WordCloud is None:
        return None
    try:
        wc = WordCloud(width=width, height=height, background_color=bg).generate(text)
        img = wc.to_image()
        bio = BytesIO()
        img.save(bio, format="PNG")
        return bio.getvalue()
    except Exception:  # pragma: no cover
        return None
