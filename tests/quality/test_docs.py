from scripts.check_docs import DOCUMENTATION_ROOTS, validate_docs


def test_documentation_limits_and_links() -> None:
    errors = [error for root in DOCUMENTATION_ROOTS for error in validate_docs(root)]
    assert errors == []
