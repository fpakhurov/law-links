"""Extraction of legal references from Russian text."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# The task template keeps the dictionary next to main.py: a file placed there
# (as the template's Dockerfile does) wins over the copy in data/.
_TEMPLATE_ALIASES_PATH = ROOT / "law_aliases.json"
DEFAULT_ALIASES_PATH = (
    _TEMPLATE_ALIASES_PATH if _TEMPLATE_ALIASES_PATH.exists() else ROOT / "data" / "law_aliases.json"
)
DEFAULT_CHAIN_MODEL_PATH = ROOT / "data" / "chain_hmm.json"
