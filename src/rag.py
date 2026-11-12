import os
import json
import zipfile

from src.indexer import indexing
from src.retriever import RetrieverEngine
from src.answerer import AnswerGeneration
from src.models import MinimalSource, StudentSearchResults
from src.evaluate import Recall
from src.utils import write_search_results, write_answer_results, RAGConfig

MAX_CONTEXT_LENGTH = 2000
EXTENSIONS = [".py", ".md", ".txt"]

end = "\033[0m"
g = "\033[32m\033[5m\033[1m"


class RAGEngine():
    """
    The Retrieval-Augmented Generation Engine. The commands and
    their flags are defined here.

    Attributes:
      - index(self, max_chunk_size: int = RAGConfig.MAX_CHUNK) -> None
      - search(self, query: str = RAGConfig.QUERY, k: int = 10,
               verbose: bool = True) -> list[MinimalSource] | None
      - search_dataset(self,
                       dataset_path: str = RAGConfig.DATASET_UNANSWERED_PATH,
                       k: int = 10,
                       save_directory: str = RAGConfig.SAVE_PATH) -> None
      - answer(self, query: str = RAGConfig.QUERY, k: int = 10) -> None
      - answer_dataset(
            self, student_search_results_path: str = RAGConfig.RESULTS_PATH,
            save_directory: str = RAGConfig.SEARCH_SAVE_PATH
            ) -> None
      - evaluate(self,
                 student_search_results_path: str = RAGConfig.RESULTS_PATH,
                 dataset_path: str = RAGConfig.DATASET_ANSWERED_PATH) -> None
    """
    def index(self, max_chunk_size: int = RAGConfig.MAX_CHUNK) -> None:
        """
        Ingest data/raw/ and build the index under data/processed/.

        Parameter:
          - max_chunk_size: int = RAGConfig.MAX_CHUNK
        """
        # Check the max_chunk_size
        if max_chunk_size <= 0:
            raise ValueError("The context length should not be "
                             "below 0 character.")
        if max_chunk_size > RAGConfig.MAX_CHUNK:
            raise ValueError("The context length should not be "
                             "above 2000 characters.")

        # Check if the required LLM folder is present
        vLLM_path: str = "data/raw/vllm-0.10.1/"
        if os.path.exists("data/raw") is False:
            os.mkdir("data/raw")

        try:
            with zipfile.ZipFile("vllm-0.10.1.zip", 'r') as zip_file:
                zip_file.extractall("data/raw")

        except Exception:
            print("No corresponding .zip file found")

        if os.path.exists(vLLM_path) is False:
            raise ValueError("Path to vllm does not exist. "
                             "Place the .zip file at the root "
                             "of the project and run again.")

        # If the folders don't exist, it is created
        if os.path.exists("data/processed") is False:
            os.mkdir("data/processed")

        if os.path.exists("data/processed/bm25") is False:
            os.mkdir("data/processed/bm25")

        print("Every missing folder has been created successfully!")

        # Index the corpus
        nb_chunks: int = indexing(max_chunk_size, vLLM_path)

        # Prints the number of chunks processed
        print(f"Ingestion complete! Indexed {nb_chunks} chunks "
              "under data/processed/")

    def search(self, query: str = RAGConfig.QUERY, k: int = 10,
               verbose: bool = True) -> list[MinimalSource] | None:
        """
        Return the top-k sources for a single query.

        Parameters:
          - query: str = RAGConfig.QUERY
          - k: int = 10
          - verbose: bool = True

        Return
        -> list[MinimalSource] | None
        """
        chunk_list: list[MinimalSource] = []

        # Check the given parameters
        if query == "":
            raise ValueError("The query cannot be empty")
        if k <= 0:
            raise ValueError("Invalid k. The value cannot be below "
                             "or equal to 0")
        if k >= RAGConfig.MAX_K:
            raise ValueError("Invalid k. The value cannot be "
                             f"higher than {RAGConfig.MAX_K}")

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

        # Write a JSON file which comply to StudentSearchResults
        write_search_results(query, chunk_list, k)

        if verbose is True:
            print(f"\nQuery: {query}\n")
            print("Sources:")
            for model in chunk_list:
                print(f"  - {model.file_path} [{model.first_character_index}:"
                      f"{model.last_character_index}]")

            print(f"\n{g}[INFO]{end}: Saved search results to "
                  "data/output/search_operations.json")
            return None

        return chunk_list

    def search_dataset(self,
                       dataset_path: str = RAGConfig.DATASET_UNANSWERED_PATH,
                       k: int = 10,
                       save_directory: str = RAGConfig.SAVE_PATH) -> None:
        """
        Run search over a whole dataset and write a
        StudentSearchResults JSON file.

        Parameters:
          - dataset_path: str = RAGConfig.DATASET_UNANSWERED_PATH
          - k: int = 10
          - save_directory: str = RAGConfig.SAVE_PATH
        """
        # Check the given parameters
        if k <= 0:
            raise ValueError("Invalid k. The value cannot be below "
                             "or equal to 0")
        if k >= RAGConfig.MAX_K:
            raise ValueError("Invalid k. The value cannot be "
                             f"higher than {RAGConfig.MAX_K}")
        if os.path.exists(dataset_path) is False:
            raise ValueError("Path to the dataset does not exist")

        # Check if the indexing was previously done
        if os.path.exists("data/processed") is False:
            raise ValueError("Indexing is not made. "
                             "Please type the command "
                             "'make index' and retry")

        # Check if the path exists
        folders = save_directory.split("/")

        if os.path.exists(save_directory) is False:
            # Check the file_path and create mising folder
            os.makedirs(save_directory)

        # Check that a file to stock the data exists
        if not folders[-1].endswith(".json"):
            file_name: str = "dataset_docs_public.json"
            if save_directory.endswith("/"):
                save_directory += file_name
            else:
                save_directory += f"/{file_name}"

        # Retrieve the top-k sources
        retriever = RetrieverEngine(k)

        dataset = retriever.create_dataset(dataset_path)
        if dataset:
            chunk_list = retriever.retrieve_dataset(dataset)
            retriever.save_dataset(chunk_list, save_directory, k)

    def answer(self, query: str = RAGConfig.QUERY, k: int = 10) -> None:
        """
        Answer a single query using the retrieved context.

        Parameters:
          - query: str = RAGConfig.QUERY
          - k: int = 10
        """
        # Check the given parameters
        if query == "":
            raise ValueError("The query cannot be empty")
        if k <= 0:
            raise ValueError("Invalid k. The value cannot be below "
                             "or equal to 0")
        if k >= RAGConfig.MAX_K:
            raise ValueError("Invalid k. The value cannot be "
                             f"higher than {RAGConfig.MAX_K}")

        # Reunite necessary resources
        search_result = self.search(query, k, False)

        if search_result is None:
            raise ValueError("No search results found")

        # Create the generator
        generator = AnswerGeneration()
        answer = generator.answer(query, search_result, None)

        # Write a JSON file which comply to StudentSearchResultsAndAnswer
        write_answer_results(query, search_result, answer, k)

        print(f"\nQuery: {query}\n")
        print(f"Answer: {answer.answer}\n")
        print("Sources:")
        for model in search_result:
            print(f"  - {model.file_path} [{model.first_character_index}:"
                  f"{model.last_character_index}]")

        print(f"\n{g}[INFO]{end}: Saved answer results to "
              "data/output/answer_operations.json")

    def answer_dataset(
            self, student_search_results_path: str = RAGConfig.RESULTS_PATH,
            save_directory: str = RAGConfig.SEARCH_SAVE_PATH
            ) -> None:
        """
        Generate answers for a dataset, producing a
        StudentSearchResultsAndAnswer JSON file.

        Parameters:
          - student_search_results_path: str = RAGConfig.RESULTS_PATH
          - save_directory: str = RAGConfig.SEARCH_SAVE_PATH
        """
        # Check if the paths exists
        if os.path.exists(student_search_results_path) is False:
            raise ValueError("Path to the student search results "
                             "does not exist")

        folders = save_directory.split("/")

        if os.path.exists(save_directory) is False:
            # Check the file_path and create mising folder
            folder_path: str = ""
            for folder in folders:
                folder_path += folder

                if os.path.exists(folder_path) is False:
                    os.mkdir(folder_path)

                folder_path += "/"

        # Check that a file to stock the data exists
        if not folders[-1].endswith(".json"):
            file_name: str = "dataset_docs_public.json"
            if save_directory.endswith("/"):
                save_directory += file_name
            else:
                save_directory += f"/{file_name}"

        generator = AnswerGeneration()

        dataset = generator.create_dataset(student_search_results_path)
        if dataset:
            answers = generator.answer_dataset(dataset)
            generator.save_dataset(dataset, answers, save_directory)

    def evaluate(self,
                 student_search_results_path: str = RAGConfig.RESULTS_PATH,
                 dataset_path: str = RAGConfig.DATASET_ANSWERED_PATH) -> None:
        """
        Report own recall@k against a ground-truth dataset,
        for own testing.

        Parameters:
          - student_search_results_path: str = RAGConfig.RESULTS_PATH
          - dataset_path: str = RAGConfig.DATASET_ANSWERED_PATH
        """
        # Check if the paths exists
        if os.path.exists(student_search_results_path) is False:
            raise ValueError("Path to the student search results "
                             "does not exist")

        if os.path.exists(dataset_path) is False:
            raise ValueError("Path to the dataset does not exist")

        recall = Recall()

        try:
            # Extract data from both path
            with open(dataset_path, "r", encoding="utf8") as f:
                dataset = json.load(f)

            with open(student_search_results_path, "r", encoding="utf8") as f:
                search_result = StudentSearchResults(**json.loads(f.read()))

        except PermissionError:
            raise ValueError(f"Permission error with this file: {f}")
        except Exception as e:
            raise ValueError(e)

        dataset_dict = {
            str(q["question_id"]): q["sources"]
            for q in dataset["rag_questions"]
        }

        result = recall.recall_k(search_result, dataset_dict)

        # Print the results
        print("🎯 Evaluation Results")
        print("========================================")
        print(f"📊 Questions evaluated: {len(dataset_dict)}")
        for recallk, score in result.items():
            print(f"📈 Recall@{recallk}: {score:.3f} ({(score * 100):.1f}%)")
