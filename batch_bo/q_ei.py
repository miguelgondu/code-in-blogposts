from pathlib import Path
import sys

from jax import config

config.update("jax_enable_x64", True)

sys.path.append(str(Path(__file__).resolve().parent))
