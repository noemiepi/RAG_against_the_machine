from transformers import pipeline

from src.models.models import (StudentSearchResultsAndAnswer, MinimalAnswer,
                               MinimalSearchResults, MinimalSource)


class AnswerGeneration():
    """
    It will generate an answer to a query or a serie of
    queries using an AI model.

    Attributes:
      -
    """
    def __init__(self) -> None:
        """
        Initialize the answer generation.
        """
        self._model = "Qwen/Qwen3-0.6B"
        self._context_limit = 3000

        self._load_llm()

    def answer(self) -> None:
        """
        .
        """
        pass

    def answer_dataset(self) -> None:
        """
        .
        """
        pass

    # --------------- #
    # Private Methods #
    # --------------- #

    def _load_llm(self) -> None:
        """
        .
        """
        # Load the model using a pipeline
        self._pipeline = pipeline(
            task="text-generation",
            model=self._model,
            device=0,
            clean_up_tokenization_spaces=False
        )

    def _generate_prompt(self) -> None:
        """
        .
        """
        pass

    def _generate_answer(self) -> None:
        """
        .
        """
        pass
