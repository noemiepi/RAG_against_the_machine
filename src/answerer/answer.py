import uuid
import tqdm

from transformers import pipeline

from src.models.models import (StudentSearchResultsAndAnswer, MinimalAnswer,
                               MinimalSearchResults, MinimalSource)

MAX_NEW_TOKENS: int = 256


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
        prompt: list[dict[str, str]] = self._generate_prompt(question,
                                                             retrived_sources)
        gen_answer: str = self._generate_answer(prompt)

        clean_answer: str = gen_answer.replace("<think>",
                                               "").replace("</think>",
                                                           "").strip()

        answer = MinimalAnswer(
            question_id=(
                str(uuid.uuid4()) if question_id is None
                else str(question_id)),
            question=question,
            retrieved_sources=retrived_sources,
            answer=clean_answer
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
        try:
            # Load the model using a pipeline
            self._pipe = pipeline(
                task="text-generation",
                model=self._model,
                device=0,
                clean_up_tokenization_spaces=False
            )

        except Exception:
            raise ValueError("Unable to load the model")

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
        formatted_sources: str = "\n".join(
            [str(source.content) for source in sources]
        )

        # Cut the text if it's too long
        if len(formatted_sources) > self._context_limit:
            formatted_sources = formatted_sources[:self._context_limit]

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
            {
                "role": "user",
                "content": f"User' prompt: '{question}'/no_think"
            }
        ]

    def _generate_answer(self, prompt: list[dict[str, str]]) -> str:
        """
        Generates an answer from a given prompt.

        Parameter:
          - prompt: list[dict[str, str]]

        Return
        -> str
        """
        output = self._pipe(prompt, return_full_text=False)
        return str(output[0]["generated_text"])
