import uuid

from src.models.models import (MinimalSource, MinimalAnswer,
                               MinimalSearchResults, StudentSearchResults,
                               StudentSearchResultsAndAnswer)


def write_search_results(query: str, retrieved_sources: list[MinimalSource],
                         k: int) -> None:
    """
    It writes the query, its id, top-k retrieved sources and k
    in a JSON file conforming to StudentSearchResults model.

    Parameters:
      - query: str
      - retrieved_sources: list[MinimalSource]
      - k: int
    """
    file_path: str = "data/output/search_operations.json"

    min_search = [MinimalSearchResults(question_id=str(uuid.uuid4()),
                                       question=query,
                                       retrieved_sources=retrieved_sources)]

    final_output = StudentSearchResults(search_results=min_search,
                                        k=k)

    # Write the model in a JSON file
    try:
        with open(file_path, "w", encoding='utf-8') as f:
            f.write(final_output.model_dump_json(indent=4))

    except PermissionError:
        raise ValueError(f"Permission error with this file: {f}")
    except Exception as e:
        raise ValueError(e)


def write_answer_results(query: str, retrieved_sources: list[MinimalSource],
                         answer: str, k: int) -> None:
    """
    It writes the query, its id, top-k retrieved sources,
    the query's answer and k in a JSON file conforming
    to StudentSearchResults model.

    Parameters:
      - query: str
      - retrieved_sources: list[MinimalSource]
      - answer: str
      - k: int
    """
    file_path: str = "data/output/answer_operations.json"

    min_answer = [MinimalAnswer(question_id=str(uuid.uuid4()),
                                question=query,
                                retrieved_sources=retrieved_sources,
                                answer=answer.answer)]

    final_output = StudentSearchResultsAndAnswer(search_results=min_answer,
                                                 k=k)

    # Write the model in a JSON file
    try:
        with open(file_path, "w", encoding='utf-8') as f:
            f.write(final_output.model_dump_json(indent=4))

    except PermissionError:
        raise ValueError(f"Permission error with this file: {f}")
    except Exception as e:
        raise ValueError(e)
