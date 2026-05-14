# Python Project Rules

<!-- Generated from pydevtools.com, the Python Developer Tooling Handbook -->
<!-- Last verified against: uv 0.11.8, ruff 0.15.12, ty 0.0.35, pytest 9.0.3 -->
<!-- Full explanations: https://pydevtools.com/handbook/explanation/modern-python-project-setup-guide-for-ai-assistants/ -->

## Package management

This project uses `uv`. Do not use `pip`, `pip-tools`, `poetry`, or `conda`.

- Add runtime dependency: `uv add <package>` (writes to `[project.dependencies]`)
- Add dev dependency: `uv add --dev <package>` (writes to `[dependency-groups]` per PEP 735)
- Remove dependency: `uv remove <package>`
- Sync environment from lockfile: `uv sync`
- Regenerate lockfile from constraints: `uv lock`
- Upgrade locked versions: `uv lock --upgrade`
- Commit `uv.lock` to version control

### Platform-specific dependencies

Platform deps live in dedicated dependency groups, not inline `sys_platform` markers.

- Linux: `uv sync --group linux`
- Windows: `uv sync --group windows`

Do not add `; sys_platform == ...` markers to `[project.dependencies]`. Put them in the `linux` or `windows` group instead.

## Running code

Always use `uv run` to execute Python code and tools. Never call `python`, `pytest`, `ruff`, or other tools directly. They may not resolve to the project's virtual environment.

- Run a script: `uv run python script.py`
- Run a module: `uv run python -m module_name`
- Run a tool: `uv run pytest`, `uv run ruff check .`
- One-off tool (not a project dependency): `uvx <tool>`

## Testing

- Framework: `pytest`
- Run tests: `uv run pytest`
- Smoke tests require the game running and are skipped by default (`-m not smoke`)
- Test files go in `tests/` at the project root
- Test file naming: `test_*.py`
- No `__init__.py` needed in `tests/`

## Linting and formatting

- Tool: `ruff` (handles both linting and formatting)
- Lint: `uv run ruff check .`
- Lint and auto-fix: `uv run ruff check --fix .`
- Format: `uv run ruff format .`
- Check formatting: `uv run ruff format --check .`
- Sort modules: `uv run ssort src/ scripts/ tests/`
- Line length: 160 characters
- All `ruff` configuration lives in `pyproject.toml` under `[tool.ruff]`

## TOML formatting

- Tool: `taplo`
- Format: `uvx taplo fmt pyproject.toml`
- Configuration lives in `pyproject.toml` under `[tool.taplo]`
- Pre-commit hook runs `taplo` automatically on staged `.toml` files

## Type checking

- Tool: `ty`
- Run: `uv run ty check`
- Configuration lives in `pyproject.toml` under `[tool.ty]`

## Tool configuration

All tool configuration lives in `pyproject.toml`. Do not create separate `ruff.toml`, `ty.toml`, `.taplo.toml`, or `pytest.ini` files.

In `[tool.uv]`, always put `[[tool.uv.index]]` above `[tool.uv.sources]` (define names before referencing them).

## Pre-commit hooks

- Tool: `prek`
- Install hooks: `uvx prek install`
- Do not install `prek` with `pip`. Use `uvx`.

## Security

- Scan dependencies for vulnerabilities: `uv audit`
- GitHub Actions pins: `scripts/pin-actions.sh` (updates SHAs via `pinact` + audits via `zizmor`)
- Check only (for CI): `scripts/pin-actions.sh --check`
- Requires `pinact` and `zizmor` installed on the system

## CI

Three separate jobs in `.github/workflows/ci.yml`:

- `lint` - `ruff check`, `ruff format`, `ty check`
- `test` - `pytest` (non-smoke)
- `audit` - `uv audit`

All GitHub Actions are SHA-pinned with trailing tag comments. Use `scripts/pin-actions.sh` to update them.

## What NOT to do

- Do not create or activate virtual environments manually. `uv` manages `.venv/` automatically.
- Do not install packages globally or with `pip install`.
- Do not create `requirements.txt` for dependency management. Use `pyproject.toml` and `uv.lock`.
- Do not run `python setup.py` commands.
- Do not add dependencies to `pyproject.toml` by hand. Use `uv add`.
- Do not create separate config files for `ruff`, `ty`, `taplo`, or `pytest`. Everything goes in `pyproject.toml`.
- If you must edit `pyproject.toml` directly, write dev dependencies under `[dependency-groups]` (PEP 735), not the legacy `[tool.uv.dev-dependencies]` table.
