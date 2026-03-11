class TrainingDataBotError(Exception):
    """Base exception for the Training Data Bot project."""


class LoaderError(TrainingDataBotError):
    """Raised when document loading fails."""


class ProcessingError(TrainingDataBotError):
    """Raised when document processing fails."""


class ExportError(TrainingDataBotError):
    """Raised when dataset export fails."""