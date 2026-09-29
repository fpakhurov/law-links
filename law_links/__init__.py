"""Extraction of legal references from Russian text."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ALIASES_PATH = ROOT / "data" / "law_aliases.json"
DEFAULT_CHAIN_MODEL_PATH = ROOT / "data" / "chain_hmm.json"
