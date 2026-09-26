.PHONY: install phoenix phoenix-docker phoenix-docker-stop api test queries export evals dataset datasets demo

install:
	poetry install

phoenix:
	PHOENIX_WORKING_DIR=$(CURDIR)/.phoenix poetry run phoenix serve

phoenix-docker:
	docker compose up -d phoenix
	@echo "Phoenix UI: http://localhost:6006"

phoenix-docker-stop:
	docker compose stop phoenix

api:
	poetry run uvicorn api:app --reload --port 8000

test:
	poetry run pytest -q

queries:
	poetry run python scripts/run_queries.py

export:
	poetry run python scripts/export_spans.py

evals:
	poetry run python scripts/run_evals.py

dataset:
	poetry run python scripts/build_dataset.py --name frustrated-interactions

datasets:
	poetry run python scripts/build_dataset.py --name frustrated-interactions
	poetry run python scripts/build_dataset.py --name hallucinated-answers
	poetry run python scripts/build_dataset.py --name tool-errors

demo: queries export evals dataset
