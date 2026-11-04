import uuid

from transformers import pipeline

from src.models.models import (StudentSearchResultsAndAnswer, MinimalAnswer,
                               MinimalSearchResults, MinimalSource)


class AnswerGeneration():
    """
    It will generate an answer to a query or a serie of
    queries using an AI model.

    Attributes:
      - answer(self, query: str, retrived_sources: list[MinimalSource],
               question_id: uuid.UUID | None) -> MinimalAnswer
      - answer_dataset(self) -> None
      - _load_llm(self) -> None
      - _generate_prompt(self, question: str,
                         sources: list[MinimalSource]) -> list[dict[str, str]]
      - _generate_answer(self, prompt: list[dict[str, str]]) -> str
    """
    def __init__(self) -> None:
        """
        Initialize the answer generation.
        """
        self._model = "Qwen/Qwen3-0.6B"
        self._context_limit = 3000

        self._load_llm()

    def answer(self, question: str, retrived_sources: list[MinimalSource],
               question_id: uuid.UUID | None) -> MinimalAnswer:
        """
        Generates an answer to a query.

        Parameters:
          - question: str
          - retrieved_sources: list[MinimalSource]
          - question_id: uuid.UUID | None

        Return
        -> MinimalAnswer
        """
        prompt: list[dict[str, str]] = self._generate_prompt()
        gen_answer: str = self._generate_answer(prompt)

        answer = MinimalAnswer(
            question_id=(
                str(uuid.uuid4()) if question_id is None
                else str(question_id)),
            question=question,
            retrieved_sources=retrived_sources,
            answer=gen_answer
        )

        return answer

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
        Load the pipeline to generate text-answers.
        """
        # Load the model using a pipeline
        self._pipeline = pipeline(
            task="text-generation",
            model=self._model,
            device=0,
            clean_up_tokenization_spaces=False
        )

    def _generate_prompt(self, question: str,
                         sources: list[MinimalSource]) -> list[dict[str, str]]:
        """
        Prepare the prompt that will be given to the pipeline.

        Parameters:
          - question: str
          - sources: list[MinimalSource]

        Return
        -> list[dict[str, str]]
        """
        # Format the sources into a form usable by the pipeline
        formatted_sources: str = ""

        # Return the prepared prompt
        # (/no_think prevents the model from using <think>)
        return [
            {
                "role": "system",
                "content": (
                    "Answer the user's prompt using only the provided"
                    "sources. Your answer will be concise but helpfull"
                    "for the user."
                    "The sources are: \n\n"
                    f"{formatted_sources}"
                ),
            },
            {"role": "user", "content": f"User' prompt: '{question}'/no_think"}
        ]

    def _generate_answer(self, prompt: list[dict[str, str]]) -> str:
        """
        Generates an answer from a given prompt.

        Parameter:
          - prompt: list[dict[str, str]]

        Return
        -> str
        """
        output = self._pipeline(prompt, return_full_text=False)
        return str(output)
