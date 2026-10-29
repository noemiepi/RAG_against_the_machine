import json

from src.models.models import (MinimalSource, MinimalAnswer,
                               MinimalSearchResults, StudentSearchResults,
                               StudentSearchResultsAndAnswer)


def write_search_results(query: str, retrieved_sources: list[str],
                         k: int) -> None:
    """
    It writes the query, its id, top-k retrieved sources and k
    in a JSON file conforming to StudentSearchResults model.

    Parameters:
      - query: str
      - retrieved_sources: list[str]
      - k: int
    """
    pass


def write_answer_results(query: str, retrieved_sources: list[str],
                         answer: str, k: int) -> None:
    """
    It writes the query, its id, top-k retrieved sources,
    the query's answer and k in a JSON file conforming
    to StudentSearchResults model.

    Parameters:
      - query: str
      - retrieved_sources: list[str]
      - answer: str
      - k: int
    """
    pass
