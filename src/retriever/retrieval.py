import os
import json
import bm25s
import Stemmer

from typing import Any

from src.models.models import (MinimalSource, MinimalSearchResults,
                               StudentSearchResults, RagDataset,
                               AnsweredQuestion, UnansweredQuestion)


class RetrieverEngine():
    """
    This is the retriever. It will group together every
    chunks to answer queries.

    Attributes:
      - retriever(self, query: str) -> list[str]
      - create_dataset(
            self, file_path: str
                       ) -> list[AnsweredQuestion | UnansweredQuestion] | None:
      - retrieve_dataset(
            self, dataset: list[AnsweredQuestion | UnansweredQuestion]
            ) -> list[MinimalSearchResults]
      - save_dataset(self, chunks: list[MinimalSearchResults],
                     file_path: str) -> None
    """
    def __init__(self, k: int) -> None:
        self.k = k

        # Reload the retriever and recreate the stemmer
        self.retriever = bm25s.BM25.load("data/processed/bm25",
                                         load_corpus=True)
        self.stemmer = Stemmer.Stemmer("english")

    def retrieve(self, query: str) -> list[MinimalSource]:
        """
        Retrieve top-k chunks to answer a query.

        Parameter:
        - query: str

        Return
        -> list[MinimalSource]
        """
        file_path: str = ""
        first: int = 0
        last: int = 0
        retrieved_sources: list[MinimalSource] = []

        # Tokenize the query
        query_token = bm25s.tokenize(query, stemmer=self.stemmer)

        # Get top-k results as a tuple doc-ids/scores
        results, scores = self.retriever.retrieve(query_token, k=self.k)

        # Write every information needed
        for i in range(self.k):
            doc = results[0, i]

            for key, value in doc.items():
                if key == "file_path":
                    file_path = value

                if key == "first_character_index":
                    first = value

                if key == "last_character_index":
                    last = value

            # Add the source to a reference list
            retrieved_sources.append(MinimalSource(file_path=file_path,
                                                   first_character_index=first,
                                                   last_character_index=last))

        return retrieved_sources

    def create_dataset(
            self, file_path: str
                       ) -> list[AnsweredQuestion | UnansweredQuestion] | None:
        """
        Check the validity of the question list.

        Parameter:
          - file_path: str

        Return
        -> list[AnsweredQuestion | UnansweredQuestion] | None
        """
        rag_dataset: list[AnsweredQuestion | UnansweredQuestion] = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                # Validate the model
                data = json.load(f)
                dataset = RagDataset.model_validate(data)
                rag_dataset = dataset.rag_questions

        except PermissionError:
            raise ValueError(f"Permission error with this file: {f}")
        except Exception as e:
            raise ValueError(e)

        return rag_dataset

    def retrieve_dataset(
            self, dataset: list[AnsweredQuestion | UnansweredQuestion]
            ) -> list[MinimalSearchResults]:
        """
        Retrieve top-k chunks to answer every query.

        Parameter:
          - dataset: list[AnsweredQuestion | UnansweredQuestion]

        Return
        -> list[MinimalSearchResults]
        """
        result_list: list[MinimalSearchResults] = []

        for data in dataset:
            result_list.append(MinimalSearchResults(
                question_id=data.question_id,
                question=data.question,
                retrieved_sources=self.retrieve(data.question)
            ))

        return result_list

    def save_dataset(self, chunks: list[MinimalSearchResults],
                     file_path: str, k: int) -> None:
        """
        Save the dataset in the wanted folder.

        Parameters:
          - chunk_list: dict[str, list[MinimalSource]]
          - file_path: str
          - k: int
        """
        final_output: dict[str, Any] = {}

        final_output = StudentSearchResults(search_results=chunks,
                                            k=k)

        # Write the model in a JSON file
        try:
            with open(file_path, "w", encoding='utf-8') as f:
                f.write(final_output.model_dump_json(indent=4))

        except PermissionError:
            raise ValueError(f"Permission error with this file: {f}")
        except Exception as e:
            raise ValueError(e)

        print(f"Saved student_search_results to {file_path}")
