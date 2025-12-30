"""Common progress reporting utilities for Streamlit pages."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import streamlit as st


@dataclass
class ProgressEvent:
    message: str
    current: int | None = None
    total: int | None = None


class ProgressReporter:
    """Simple progress helper that renders a banner + progress bar in Streamlit."""

    def __init__(self, title: str):
        self._container = st.container()
        self._title_placeholder = self._container.info(title)
        self._message_placeholder = self._container.empty()
        self._progress_placeholder = self._container.progress(0.0)
        self._total: int | None = None
        self._current: int = 0
        self._finalized: bool = False

    def callback(self, message: str, current: int | None = None, total: int | None = None) -> None:
        """Signature compatible with IssueService progress callbacks."""
        self.update(message, current=current, total=total)

    def update(self, message: str, *, current: int | None = None, total: int | None = None) -> None:
        if self._finalized:
            return
        if total is not None and total > 0:
            self._total = total
        if current is not None:
            self._current = max(0, current)
        self._message_placeholder.write(message)
        self._refresh_progress()

    def complete(self, message: str) -> None:
        if self._finalized:
            return
        self._progress_placeholder.progress(1.0)
        self._container.success(message)
        self._finalized = True

    def error(self, message: str) -> None:
        if self._finalized:
            return
        self._container.error(message)
        self._finalized = True

    def _refresh_progress(self) -> None:
        if self._total and self._total > 0:
            ratio = min(max(self._current / self._total, 0.0), 1.0)
            self._progress_placeholder.progress(ratio)
        else:
            # Unknown total; show indeterminate state by resetting to 0.
            self._progress_placeholder.progress(0.0)


ProgressCallback = Callable[[str, int | None, int | None], None]
