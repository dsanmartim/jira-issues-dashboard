"""Test configuration ensuring local package import when editable install not active.

If users invoke `pytest` outside the project's virtualenv, we still add the project
root to sys.path so `import jira_app` works.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
