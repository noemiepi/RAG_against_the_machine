import json
import bm25s
import Stemmer

from src.models.models import (MinimalSource, RagDataset,
                               AnsweredQuestion, UnansweredQuestion)
from src.utils.search_answer_utils import write_search_results


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
            ) -> list[MinimalSource]
      - save_dataset(self, chunk_list: list[MinimalSource],
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

        # Write a JSON file which comply to StudentSearchResults
        write_search_results(query, retrieved_sources, self.k)

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

        with open(file_path, "r", encoding="utf-8") as f:
            # Validate the model
            data = json.load(f)
            dataset = RagDataset.model_validate(data)
            rag_dataset.append(dataset)

        return rag_dataset

    def retrieve_dataset(
            self, dataset: list[AnsweredQuestion | UnansweredQuestion]
            ) -> list[MinimalSource]:
        """
        Retrieve top-k chunks to answer every query.

        Parameter:
          - dataset: list[AnsweredQuestion | UnansweredQuestion]

        Return
        -> list[MinimalSource]
        """
        pass

    def save_dataset(self, chunk_list: list[MinimalSource],
                     file_path: str) -> None:
        """
        Save the dataset in the wanted folder.

        Parameters:
          - chunk_list: list[MinimalSource]
          - file_path: str
        """
        pass
