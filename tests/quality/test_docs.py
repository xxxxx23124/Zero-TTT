from pathlib import Path

from scripts.check_docs import validate_docs


def test_documentation_limits_and_links() -> None:
    assert validate_docs(Path("docs")) == []
