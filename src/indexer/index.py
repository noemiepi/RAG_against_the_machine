import json
import bm25s
import Stemmer

from tqdm import tqdm
from typing import Any

from src.indexer.chunker import Chunker
from src.utils.file_utils import load_files


def indexing(max_chunk_size: int, vLLM_path: str) -> int:
    """
    Index a corpus and creates a JSON file reuniting every
    chunks with their name, their start and finish point.
    It returns the number of chunks created.

    Parameters:
      - max_chunk_size: int
      - vLLM_path: str

    Return
    -> int
    """
    metadata_list: list[dict[str, Any]] = []
    text_list: list[str] = []

    # Load the files
    file_dict: dict[str, str] = load_files(vLLM_path)

    # Initialize the Chunker
    chunking = Chunker(max_chunk_size)

    # Chunk each file one by one and writes it in a JSON file
    for file, content in tqdm(file_dict.items(), unit=" file",
                              desc="Chunking"):
        list_info = chunking.create_chunks(file, content)

        if list_info:
            for metadata in list_info:
                metadata_list.append(metadata.model_dump())
                text_list.append(metadata.content)

    nb_chunk: int = create_source_info(metadata_list)

    bm25_index(text_list, metadata_list)

    return nb_chunk


def create_source_info(metadata_list: list[dict[str, Any]]) -> int:
    """
    Create a JSON file with every chunk given
    and return how many there are.

    Parameter:
      - metadata_list: list[dict[str, Any]]

    Return
    -> int
    """
    json_file: list[dict[str, Any]] = []

    for lst in tqdm(metadata_list, unit=" chunk", desc="Tokenizing"):
        json_file.append(
            {
                "file_path": lst["file_path"],
                "first_character_index": lst["first_character_index"],
                "last_character_index": lst["last_character_index"],
                "content": lst["content"]
            }
        )

    try:
        with open("data/processed/chunks.json", "w", encoding='utf-8') as f:
            json.dump(json_file, f, indent=4)

    except PermissionError:
        raise ValueError(f"Permission error with this file: {f}")
    except Exception as e:
        raise ValueError(e)

    return len(json_file)


def bm25_index(texts: list[str], corpus: list[dict[str, Any]]) -> None:
    """
    Creates an index for the bm25 retrieval.

    Parameters:
      - texts: list[str]
      - corpus: list[dict[str, Any]]
    """
    # Create a stemmer
    stemmer = Stemmer.Stemmer("english")

    # Create the BM25 model
    retriever = bm25s.BM25()

    # Tokenize the corpus
    corpus_token = bm25s.tokenize(texts, stopwords="en", stemmer=stemmer)

    # Index the corpus and save the array to a directory
    retriever.index(corpus_token)

    # Save the corpus with the model to be used elsewhere
    retriever.save("data/processed/bm25", corpus=corpus)
