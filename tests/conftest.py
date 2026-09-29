import pytest

from law_links import DEFAULT_ALIASES_PATH
from law_links.aliases import AliasIndex
from law_links.extractor import RuleBasedExtractor


@pytest.fixture(scope="session")
def index() -> AliasIndex:
    return AliasIndex.from_json(DEFAULT_ALIASES_PATH)


@pytest.fixture(scope="session")
def extractor(index) -> RuleBasedExtractor:
    return RuleBasedExtractor(index)
