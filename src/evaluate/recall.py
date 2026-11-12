from typing import Any

from src.models import (MinimalSource, MinimalSearchResults,
                        StudentSearchResults)


class Recall():
    """
    Evaluate the text retrieval and its relevance by comparing
    it to a series of questions that already have a top-k sources.

    Attributes:
      - recall_k(self, search_results: StudentSearchResults,
                 dataset: dict[str, Any]) -> None
      - _valid_question(self, search_result: MinimalSearchResults,
                        expected_src: list[dict[str, Any]], k: int) -> int
      - _compute_iou(source_a: MinimalSource,
                     source_b: MinimalSource) -> float
    """
    def recall_k(self, search_results: StudentSearchResults,
                 dataset: dict[str, Any]) -> dict[str, float]:
        """
        Compute recall@k scores for the provided search results.

        Parameters:
          - search_results: StudentSearchResults
          - dataset: dict[str, Any]

        Return:
        -> dict[str, float]
        """
        recalls = {"1": 0.0, "3": 0.0, "5": 0.0, "10": 0.0}

        for k in recalls.keys():
            if (int(k) <= search_results.k):
                validated = 0

                for search_result in search_results.search_results:
                    src = dataset.get(search_result.question_id)

                    if src is not None:
                        validated += self._valid_question(search_result,
                                                          src, int(k))

                recalls[k] = (validated / len(dataset)
                              if len(dataset) > 0 else 0)

        return recalls

    # --------------- #
    # Private Methods #
    # --------------- #

    def _valid_question(self, search_result: MinimalSearchResults,
                        expected_src: list[dict[str, Any]], k: int) -> int:
        """
        Check if a at least one retrieved source matches the answer.

        Parameters:
          - search_result: MinimalSearchResults
          - expected_src: list[dict[str, Any]]
          - k: int

        Return:
        -> int
        """
        for chunk in expected_src:
            for minimal_src in search_result.retrieved_sources[:k]:
                if (minimal_src.file_path == chunk['file_path'] and
                   self._compute_iou(minimal_src, chunk) >= 0.05):
                    return 1
        return 0

    def _compute_iou(self, source_a: MinimalSource,
                     source_b: dict[str, Any]) -> float:
        """
        Calculate the IoU between two source spans.

        Parameters:
        - source_a: MinimalSource
        - source_b: MinimalSource

        Return
        -> float
        """
        # Extract the sources
        inter_start: int = max(source_a.first_character_index,
                               source_b['first_character_index'])
        inter_end: int = min(source_a.last_character_index,
                             source_b['last_character_index'])

        # Compute intersection
        inter: int = max(0, inter_end - inter_start)

        # Compute both boxes (intersection and union)
        len_a: int = max(
            0, source_a.last_character_index - source_a.first_character_index
            )
        len_b: int = max(0, (source_b['last_character_index'] -
                             source_b['first_character_index']))

        # Compute union
        union: int = len_a + len_b - inter

        # Compute IoU
        if union <= 0:
            return 0.0

        iou: float = inter / union

        return iou
