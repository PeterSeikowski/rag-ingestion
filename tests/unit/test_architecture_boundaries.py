"""Static enforcement of the architecture boundary rules that
test_domain_models.py's `test_domain_package_has_no_infrastructure_imports`
doesn't cover: application/ must also stay infra-free, and each
single-adapter-only infra library (elasticsearch/docling/litellm/redis)
must be importable from exactly the one adapter file that's supposed to
own it — a regression here (e.g. `import redis` sneaking into
application/job_service.py) would otherwise pass silently forever, since
nothing else in the test suite checks import boundaries this broadly.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "rag_ingestion"

_FORBIDDEN_IN_APPLICATION = {"fastapi", "celery", "elasticsearch", "docling", "litellm", "redis"}

_SINGLE_ADAPTER_LIBRARIES = {
    "elasticsearch": {"adapters/vectorstores/elasticsearch_writer.py"},
    "docling": {"adapters/parsers/docling_pdf_parser.py"},
    "litellm": {"adapters/embedders/litellm_embedder.py"},
    "redis": {"adapters/jobs/redis_job_repository.py"},
}


def _top_level_imports(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".")[0])
    return found


def _all_source_files() -> list[pathlib.Path]:
    return sorted(SRC.rglob("*.py"))


def test_application_package_has_no_infrastructure_imports():
    application_dir = SRC / "application"
    assert application_dir.is_dir()

    offenders: dict[str, set[str]] = {}
    for path in application_dir.glob("*.py"):
        hit = _top_level_imports(path) & _FORBIDDEN_IN_APPLICATION
        if hit:
            offenders[path.name] = hit

    assert not offenders, f"application/ files import forbidden infrastructure: {offenders}"


@pytest.mark.parametrize("library,allowed_files", sorted(_SINGLE_ADAPTER_LIBRARIES.items()))
def test_single_adapter_library_imported_only_from_its_owning_file(library, allowed_files):
    offenders = []
    for path in _all_source_files():
        relative = path.relative_to(SRC).as_posix()
        if relative in allowed_files:
            continue
        if library in _top_level_imports(path):
            offenders.append(relative)

    assert not offenders, f"{library!r} is imported outside its designated adapter file: {offenders}"


def test_fastapi_only_imported_under_api_layer():
    offenders = []
    for path in _all_source_files():
        relative = path.relative_to(SRC).as_posix()
        if relative.startswith("api/"):
            continue
        if "fastapi" in _top_level_imports(path):
            offenders.append(relative)

    assert not offenders, f"fastapi is imported outside api/: {offenders}"


def test_celery_only_imported_under_workers_layer():
    offenders = []
    for path in _all_source_files():
        relative = path.relative_to(SRC).as_posix()
        if relative.startswith("workers/"):
            continue
        if "celery" in _top_level_imports(path):
            offenders.append(relative)

    assert not offenders, f"celery is imported outside workers/: {offenders}"
