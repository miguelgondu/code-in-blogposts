EXAMPLES_DIR := src/batch_bo/examples
EXAMPLES := $(wildcard $(EXAMPLES_DIR)/*.py)
FIGURES_DIR := figures
FFMPEG_BASE_CMD := ffmpeg -y -vf "fps=2,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -vcodec libx264 -pix_fmt yuv420p

all: \
	install
    plot_initial_figure \
	plot_initial_fit \
	batch_ts \
	batch_ts_video \
	qEI \
	qEI_video

install:
	uv sync

plot_initial_figure: $(EXAMPLES_DIR)/plot_initial_figure.py
	uv run python $^

plot_initial_fit: $(EXAMPLES_DIR)/plot_initial_fit.py
	uv run python $^

batch_ts: $(EXAMPLES_DIR)/batch_ts.py
	uv run python $^

batch_ts_video: batch_ts
	$(FFMPEG_BASE_CMD) \
	-i $(FIGURES_DIR)/batch_ts/batch_ts_%02d.jpg \
	$(FIGURES_DIR)/batch_ts/batch_ts.mp4

qEI: $(EXAMPLES_DIR)/qEI.py
	uv run python $^

qEI_video:
	$(FFMPEG_BASE_CMD) \
	-i $(FIGURES_DIR)/qEI/qEI_%02d.jpg \
	$(FIGURES_DIR)/qEI/qEI.mp4

.PHONY: \
	all \
	plot_initial_figure \
	plot_initial_fit \
	batch_ts \
	batch_ts_video \
	qEI \
	qEI_video
