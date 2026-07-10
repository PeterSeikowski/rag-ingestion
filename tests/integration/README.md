# Integration tests

Empty by design for this MVP — see `tests/unit/` for the full test suite, which covers every adapter's behavior against hand-built fakes (no real infrastructure needed).

Real integration tests belong here once the following are available:

- **Redis + Elasticsearch**: `docker compose up redis` plus a local Elasticsearch 8.x instance (see the main README's "Local setup" section). Mark tests `@pytest.mark.integration`; they're excluded from the default run.
- **Docling**: `pip install docling==<pinned version>` (see `requirements.txt` — deliberately not installed by default; it pulls in torch and is slow to install). Mark tests `@pytest.mark.docling_integration` and use them to re-verify the field-extraction assumptions documented with `# TODO(docling-api)` comments in `adapters/parsers/docling_pdf_parser.py` against the real installed version.
- **LiteLLM**: a real (or sandboxed test) API key for whatever provider `LITELLM_MODEL` points at.

Both markers are pre-registered in `pytest.ini`. Run everything including integration tests with:

```bash
pytest -m "integration or docling_integration"
```

The default `pytest` invocation (see root README) runs neither.
