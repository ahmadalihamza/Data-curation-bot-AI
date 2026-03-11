"""
Training Data Curation Bot
"""

from training_data_bot.bot import TrainingDataBot
from training_data_bot.core.config import settings
from training_data_bot.sources.unified import UnifiedLoader

__version__ = "0.1.0"
__author__ = "Training Data Bot Team"
__description__ = "Enterprise-grade training data curation bot for LLM fine-tuning"

__all__ = [
    "TrainingDataBot",
    "settings",
    "UnifiedLoader",
]