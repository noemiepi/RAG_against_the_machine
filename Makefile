# ----------------- #
#     Variables     #
# ----------------- #

PYTHON=uv run python

FLAKE8=uv run flake8
MYPY=uv run mypy

CHECK_UV=command -v uv
INSTALL_UV = curl -LsSf https://astral.sh/uv/install.sh | sh

MYPY_FLAGS=--warn-return-any --warn-unused-ignores --ignore-missing-imports \
--disallow-untyped-defs --check-untyped-defs \

STUDENT_SEARCH_RESULTS_PATH = data/output/search_results/UnansweredQuestions/dataset_docs_public.json

# ----------------- #
#       Rules       #
# ----------------- #

.PHONY: all install run debug clean lint lint-strict lint-format index search


all: install run


install:
	@clear
	@if	! $(CHECK_UV) > /dev/null 2>&1; then \
			echo "UV not installed. Installing..."; \
			$(INSTALL_UV); \
	fi
	@echo "$(BROWN)Installing project dependencies using uv...$(END)"
	uv sync --link-mode=copy


run:
	@clear
	@echo "$(BLUE)Running the project...$(END)"
	$(PYTHON) -m src


debug:
	@clear
	@echo "$(BLUE)Running the project in debug...$(END)"
	$(PYTHON) -m pdb src $(JSON_FLAGS)


clean:
	@clear
	@echo "$(RED)Removing unecessary files from the folder...$(END)"
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf data/processed
	rm -rf data/output
	@echo "\n$(GREEN)Folder cleaned!$(END)"


lint:
	@clear
	@status=0; \
	$(FLAKE8) src/ || status=$$?; \
	$(MYPY) src/ $(MYPY_FLAGS) || status=$$?; \
	exit $$status


lint-strict:
	@clear
	@status=0; \
	$(FLAKE8) src/ || status=$$?; \
	$(MYPY) src/ $(MYPY_FLAGS) --strict || status=$$?; \
	exit $$status

# ----------------- #
#    Bonus Rules    #
# ----------------- #

index: install
	   @clear
	   @echo "$(BLUE)Indexing the files...$(END)"
	   $(PYTHON) -m src index --max_chunk_size 2000


search:
	@clear
	@echo "$(BLUE)Retrieving the corresponding files to answer "How to configure the OpenAI server?"...$(END)"
	$(PYTHON) -m src search "How to configure the OpenAI server?" --k 5


search_dataset:
	@clear
	@echo "$(BLUE)Searching a dataset...$(END)"
	$(PYTHON) -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --k 10 --save_directory data/output/search_results/UnansweredQuestions

answer:
	@clear
	@echo "$(BLUE)Answering "How to configure the OpenAI server?"...$(END)"
	$(PYTHON) -m src answer "How to configure the OpenAI server?" --k 5


answer_dataset:
	@clear
	@echo "$(BLUE)Answering a dataset...$(END)"
	$(PYTHON) -m src answer_dataset --student_search_results_path ${STUDENT_SEARCH_RESULTS_PATH} --save_directory data/output/search_results_and_answer/UnansweredQuestions


evaluate:
	@clear
	@echo "$(BLUE)Evaluating the recall@k...$(END)"
	$(PYTHON) -m src evaluate --student_search_results_path ${STUDENT_SEARCH_RESULTS_PATH} --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json

# ----------------- #
#       Colors      #
# ----------------- #

BROWN=\e[1;33m
BLUE=\e[1;94m
RED=\e[1;31m
GREEN=\e[1;92m
END=\e[0m
