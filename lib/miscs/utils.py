from pathlib import Path

from .logutils import logger_manager
logger = logger_manager.register(__name__)

projectdir = Path(__file__).resolve().parents[2]
assetsdir = projectdir / 'results'
datadir = projectdir / 'results'

eps = 1e-12
