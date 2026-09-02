from scripts.check_docs import DOCUMENTATION_ROOTS, validate_docs
from scripts.generate_contracts import generate


def test_documentation_limits_and_links() -> None:
    errors = [error for root in DOCUMENTATION_ROOTS for error in validate_docs(root)]
    assert errors == []


def test_openapi_and_json_schemas_are_generated_from_code() -> None:
    assert generate(check=True) == []
