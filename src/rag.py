import os

from src.indexer.index import indexing

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

        # If the folder doesn't exist, it is created
        if os.path.exists("data/processed/") is False:
            os.mkdir("data/processed")
            print("data/processed/ folder has been created successfully!")

        # Index the corpus
        nb_chunks: int = indexing(max_chunk_size, vLLM_path)

        # Prints the number of chunks processed
        print(f"Ingestion complete! Indexed {nb_chunks} chunks "
              "under data/processed/")

    def search(self, query: str, k: int) -> str:
        """
        Return the top-k sources for a single query.

        Parameters:
          - query: str
          - k: int

        Return
        -> str
        """
        pass

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
        pass

    def answer(self, query: str, k: int) -> None:
        """
        Answer a single query using the retrieved context.

        Parameters:
          - query: str
          - k: int
        """
        pass

    def answer_dataset(self, student_search_results_path: str,
                       save_directory: str) -> None:
        """
        Generate answers for a dataset, producing a
        StudentSearchResultsAndAnswer JSON file.

        Parameters:
          - student_search_results_path: str
          - save_directory: str
        """
        pass

    def evaluate(self, student_search_results_path: str,
                 dataset_path: str) -> None:
        """
        Report your own recall@k against a ground-truth
        dataset, for your own testing.

        Parameters:
          - student_search_results_path: str
          - dataset_path: str
        """
        pass
