import json
import bm25s
import Stemmer

from tqdm import tqdm

from src.indexer.chunker import Chunker
from src.utils.file_utils import load_files
from src.models.models import MinimalSource


def indexing(max_chunk_size: int, vLLM_path: str) -> int:
    """
    Index a corpus.

    Parameters:
      - max_chunk_size: int
      - vLLM_path: str

    Return
    -> int
    """
    # Load the files
    file_dict: dict[str, str] = load_files(vLLM_path)
    dict_info: dict[str, list[tuple[MinimalSource, str]]] = {}

    # Initialize the Chunker
    chunking = Chunker(max_chunk_size)

    for file, content in tqdm(file_dict.items(), unit=" file",
                              desc="Chunking"):
        print(file)
        dict_info[file] = chunking.create_chunks(file, content)

    nb_chunk: int = create_source_info(dict_info)

    return nb_chunk

def create_source_info(dict_info: dict[str, str]) -> int:
    json_file: list[dict[str, str]] = []

    # with open("source_information", "w") as:
