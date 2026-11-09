import uuid
import json

from tqdm import tqdm
from transformers import pipeline

from src.models.models import (StudentSearchResultsAndAnswer, MinimalAnswer,
                               StudentSearchResults, MinimalSource)

MAX_NEW_TOKENS: int = 256


end = "\033[0m"
g = "\033[32m\033[5m\033[1m"


class AnswerGeneration():
    """
    It will generate an answer to a query or a serie of
    queries using an AI model.

    Attributes:
      - answer(self, query: str,
               retrived_sources: list[MinimalSource] | None,
               question_id: uuid.UUID | None) -> MinimalAnswer
      - answer_dataset(self,
                       dataset: StudentSearchResults) -> list[MinimalAnswer]
      - create_dataset(self, file_path: str) -> StudentSearchResults | None
      - save_dataset(self, search_results: StudentSearchResults,
                     answers: list[MinimalAnswer],
                     file_path: str) -> StudentSearchResultsAndAnswer
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

    def answer(self, question: str,
               retrived_sources: list[MinimalSource] | None,
               question_id: uuid.UUID | None) -> MinimalAnswer:
        """
        Generates an answer to a query.

        Parameters:
          - question: str
          - retrieved_sources: list[MinimalSource] | None
          - question_id: uuid.UUID | None

        Return
        -> MinimalAnswer
        """
        if retrived_sources is None:
            raise ValueError("No retrieved sources provided")

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

    def answer_dataset(self,
                       dataset: StudentSearchResults) -> list[MinimalAnswer]:
        """
        Generate an answer for a dataset.

        Parameter:
          - dataset: StudentSearchResult

        Return
        -> list[MinimalAnswer]
        """
        answers_list: list[MinimalAnswer] = []

        for query in tqdm(dataset.search_results, unit=" query",
                          desc="Answering"):
            answer = self.answer(query.question, query.retrieved_sources,
                                 query.question_id)

            answers_list.append(answer)

        print(f"\n{g}[INFO]{end}: Processed {len(answers_list)} of "
              f"{len(dataset.search_results)} questions")

        return answers_list

    def create_dataset(self, file_path: str) -> StudentSearchResults | None:
        """
        Check the validity of the question list.

        Parameter:
            - file_path: str

        Return
        -> StudentSearchResults | None
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                # Validate the model
                data = json.load(f)
                dataset = StudentSearchResults.model_validate(data)

        except PermissionError:
            raise ValueError(f"Permission error with this file: {f}")
        except Exception as e:
            raise ValueError(e)

        print(f"\n{g}[INFO]{end}: Loaded {len(dataset.search_results)} "
              "questions...")

        return dataset

    def save_dataset(self, search_results: StudentSearchResults,
                     answers: list[MinimalAnswer], file_path: str) -> None:
        """
        Save the dataset in the wanted folder.

        Parameters:
          - search_results: StudentSearchResults
          - answers: list[MinimalAnswer]
          - file_path: str
        """
        final_output = StudentSearchResultsAndAnswer(search_results=answers,
                                                     k=search_results.k)

        # Write the model in a JSON file
        try:
            with open(file_path, "w", encoding='utf-8') as f:
                f.write(final_output.model_dump_json(indent=4))

        except PermissionError:
            raise ValueError(f"Permission error with this file: {f}")
        except Exception as e:
            raise ValueError(e)

        print(f"\n{g}[INFO]{end}: Saved student_search_results_and_answer "
              f"to {file_path}")

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
