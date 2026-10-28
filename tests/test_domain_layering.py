import ast
import pathlib

DOMAIN = pathlib.Path("app/domain")
FORBIDDEN_PREFIXES = ("app.application", "app.infrastructure", "app.api", "app.config",
                      "app.main", "app.bootstrap")
FORBIDDEN_TOP = ("fastapi", "pydantic", "sqlalchemy", "cv2", "qdrant_client", "insightface",
                 "onnxruntime", "pydantic_settings")


def _imported_modules(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
    return names


def test_domain_package_exists():
    assert (DOMAIN / "errors.py").is_file()


def test_domain_imports_nothing_outward():
    offenders = []
    for path in DOMAIN.rglob("*.py"):
        for module in _imported_modules(path):
            if module.startswith(FORBIDDEN_PREFIXES) or module.split(".")[0] in FORBIDDEN_TOP:
                offenders.append(f"{path}: {module}")
    assert offenders == [], "domain layer reaches outward:\n" + "\n".join(offenders)
