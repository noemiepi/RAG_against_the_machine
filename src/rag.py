import os

from src.indexer.index import indexing
from src.retriever.retrieval import RetrieverEngine
from src.models.models import (MinimalSource, AnsweredQuestion,
                               UnansweredQuestion)

MAX_CONTEXT_LENGTH = 2000
EXTENSIONS = [".py", ".md", ".txt"]


class RAGEngine():
    """
    The Retrieval-Augmented Generation Engine. The commands and
    their flags are defined here.

    Attributes:
      - index(self, max_chunk_size: int) -> None
      - search(self, query: str, k: int) -> str
      - search_dataset(self, dataset_path: str, k: int,
                       save_directory: str) -> None
      - answer(self, query: str, k: int) -> None
      - answer_dataset(self, student_search_results_path: str,
                       save_directory: str) -> None
      - evaluate(self, student_search_results_path: str,
                 dataset_path: str) -> None
    """
    def index(self, max_chunk_size: int) -> None:
        """
        Ingest data/raw/ and build the index under data/processed/.

        Parameter:
          - max_chunk_size: int
        """
        # Check the max_chunk_size
        if max_chunk_size <= 0:
            raise ValueError("The context length should not be "
                             "below 0 character.")
        if max_chunk_size > 2000:
            raise ValueError("The context length should not be "
                             "above 2000 characters.")

        # Check if the required folder is present
        vLLM_path: str = "data/raw/vllm-0.10.1/"
        if os.path.exists(vLLM_path) is False:
            raise ValueError("Path to vllm does not exist")

        # If the folders don't exist, it is created
        if os.path.exists("data/processed") is False:
            os.mkdir("data/processed")

            if os.path.exists("data/processed/chunks") is False:
                os.mkdir("data/processed/chunks")

            if os.path.exists("data/processed/bm25") is False:
                os.mkdir("data/processed/bm25")

            print("Every missing folder has been created successfully!")

        # Index the corpus
        nb_chunks: int = indexing(max_chunk_size, vLLM_path)

        # Prints the number of chunks processed
        print(f"Ingestion complete! Indexed {nb_chunks} chunks "
              "under data/processed/")

    def search(self, query: str, k: int) -> list[MinimalSource]:
        """
        Return the top-k sources for a single query.

        Parameters:
          - query: str
          - k: int

        Return
        -> list[MinimalSource]
        """
        chunk_list: list[MinimalSource] = []

        # Check the given parameters
        if query == "":
            raise ValueError("The query cannot be empty")
        if k <= 0:
            raise ValueError("Invalid k. The value cannot be below "
                             "or equal to 0")

        # Check if the indexing was previously done
        if os.path.exists("data/processed") is False:
            raise ValueError("Indexing is not made. "
                             "Please type the command "
                             "'make index' and retry")

        # Check if the folder exists
        if os.path.exists("data/output") is False:
            os.mkdir("data/output")

        # Retrieve the top-k sources
        retriever = RetrieverEngine(k)

        chunk_list = retriever.retrieve(query)

        for model in chunk_list:
            print(f"{model.file_path} [{model.first_character_index}:"
                  f"{model.last_character_index}]")

        return chunk_list

    def search_dataset(self, dataset_path: str, k: int,
                       save_directory: str) -> None:
        """
        Run search over a whole dataset and write a
        StudentSearchResults JSON file.

        Parameters:
          - dataset_path: str
          - k: int,
          - save_directory: str
        """
        file_name: str = "dataset_docs_public.json"
        chunk_list: list[MinimalSource] = []

        # Check the given parameters
        if k <= 0:
            raise ValueError("Invalid k. The value cannot be below "
                             "or equal to 0")
        if os.path.exists(dataset_path) is False:
            raise ValueError("Path to the dataset does not exist")

        # Check if the path exists
        if os.path.exists(save_directory) is False:
            print("Path doesn't exist yet")

        # Retrieve the top-k sources
        retriever = RetrieverEngine(k)

        dataset = retriever.create_dataset(dataset_path)
        if dataset:
            chunk_list = retriever.retrieve_dataset()
            retriever.save_dataset()

        print(f"Saved student_search_results to {save_directory}/{file_name}")
        print("work in progress")

    def answer(self, query: str, k: int) -> None:
        """
        Answer a single query using the retrieved context.

        Parameters:
          - query: str
          - k: int
        """
        # Check the given parameters
        if query == "":
            raise ValueError("The query cannot be empty")
        if k <= 0:
            raise ValueError("Invalid k. The value cannot be below "
                             "or equal to 0")

        print("work in progress")

    def answer_dataset(self, student_search_results_path: str,
                       save_directory: str) -> None:
        """
        Generate answers for a dataset, producing a
        StudentSearchResultsAndAnswer JSON file.

        Parameters:
          - student_search_results_path: str
          - save_directory: str
        """
        print("work in progress")

    def evaluate(self, student_search_results_path: str,
                 dataset_path: str) -> None:
        """
        Report your own recall@k against a ground-truth
        dataset, for your own testing.

        Parameters:
          - student_search_results_path: str
          - dataset_path: str
        """
        print("work in progress")
