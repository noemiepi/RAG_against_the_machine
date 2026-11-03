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
    min_search = [MinimalSearchResults(question_id="q1",
                                       question=query,
                                       retrieved_sources=retrieved_sources)]

    final_output = StudentSearchResults(search_results=min_search,
                                        k=k)

    # Write the model in a JSON file
    try:
        with open("data/output/search_operations.json",
                  "w", encoding='utf-8') as f:
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
    pass
