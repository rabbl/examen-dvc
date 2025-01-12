.SILENT:
.PHONY: help

done = printf "\e[32m ✔ Done\e[0m\n\n";

## This help screen
help:
	printf "Available commands\n\n"
	awk '/^[a-zA-Z\-\_0-9]+:/ { \
		helpMessage = match(lastLine, /^## (.*)/); \
		if (helpMessage) { \
			helpCommand = substr($$1, 0, index($$1, ":")-1); \
			helpMessage = substr(lastLine, RSTART + 3, RLENGTH); \
			printf "\033[33m%-40s\033[0m %s\n", helpCommand, helpMessage; \
		} \
	} \
	{ lastLine = $$0 }' $(MAKEFILE_LIST)

PROJECT = datascientest
VENV_PATH = .venv

.activate:
	test -d "$(VENV_PATH)" || python -m venv "$(VENV_PATH)"

## Install
install: .activate
	pip install -r requirements.txt
	$(done)
.PHONY: install

## Update project
update: .activate
	echo "Updating project"
	pip install --upgrade -r requirements.txt
	$(done)
.PHONY: update

## Split data
split-data:
	python src/data/split_data.py
	$(done)
.PHONY: split-data

## Normalize data
normalize-data:
	python src/data/normalize_data.py
	$(done)
.PHONY: normalize-data

## Grid search
grid-search:
	python src/models/grid_search.py
	$(done)
.PHONY: grid-search

## Train model
train-model:
	python src/models/train_model.py
	$(done)
.PHONY: train-model

## Evaluate model
evaluate-model:
	python src/models/evaluate_model.py
	$(done)
.PHONY: evaluate-models



