import pytest

from law_links import DEFAULT_ALIASES_PATH, DEFAULT_CHAIN_MODEL_PATH
from law_links.extractor import Extractor


@pytest.fixture(scope="session")
def extractor() -> Extractor:
    return Extractor.from_files(DEFAULT_ALIASES_PATH, DEFAULT_CHAIN_MODEL_PATH)


@pytest.fixture(scope="session")
def resolver(extractor):
    return extractor.resolver


@pytest.fixture(scope="session")
def chains(extractor):
    return extractor.chains
