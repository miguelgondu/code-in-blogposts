EXAMPLES_DIR := src/batch_bo/examples
EXAMPLES := $(wildcard $(EXAMPLES_DIR)/*.py)

all: run_examples plot_initial_figure plot_initial_fit

install:
	uv sync

plot_initial_figure: $(EXAMPLES_DIR)/plot_initial_figure.py
	uv run python $^

plot_initial_fit: $(EXAMPLES_DIR)/plot_initial_fit.py
	uv run python $^


.PHONY: all plot_initial_figure plot_initial_fit
