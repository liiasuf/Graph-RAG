.PHONY: setup fmt lint typecheck test precommit hooks clean

setup:
	uv venv
	uv pip install --upgrade pip
	uv pip install pre-commit ruff black mypy pytest pytest-cov
	uv run pre-commit install

fmt:
	uv run ruff check --fix .
	uv run ruff format .

lint:
	uv run ruff check .

typecheck:
	uv run mypy .

test:
	uv run pytest -q --disable-warnings

precommit:
	uv run pre-commit run --all-files

hooks:
	uv run pre-commit install

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache
