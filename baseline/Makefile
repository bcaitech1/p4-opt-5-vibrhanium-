format:
	black . --exclude data
	isort . --skip data

lint_test:
	black . --exclude data
	isort . --skip data
	env PYTHONPATH=. pytest -vv --pylint --flake8 --ignore=data --ignore=config

test:
	env PYTHONPATH=. pytest tests --cov=src --cov-report term-missing --cov-report html
