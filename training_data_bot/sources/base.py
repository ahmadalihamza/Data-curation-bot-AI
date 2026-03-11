from training_data_bot.core.logging import get_logger


class BaseLoader:
    def __init__(self) -> None:
        self.logger = get_logger(self.__class__.__name__)

    async def load_single(self, source: str):
        raise NotImplementedError("Subclasses must implement load_single().")