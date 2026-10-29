import bm25s
import Stemmer


def retriever(query: str, k: int) -> list[str]:
    """
    Retrieve top-k sources to answer a query.

    Parameters:
      - query: str
      - k: int

    Return
    -> list[str]
    """
    # Reload the retriever and recreate the stemmer
    retriever = bm25s.BM25.load("data/processed/bm25", load_corpus=True)

    stemmer = Stemmer.Stemmer("english")

    # Tokenize the query
    query_token = bm25s.tokenize(query, stemmer=stemmer)

    # Get top-k results as a tuple doc-ids/scores
    results, scores = retriever.retrieve(query_token, k=k)

    file_path: str = ""
    first: int = 0
    last: int = 0
    ref_list: list[str] = []

    for i in range(k):
        doc = results[0, i]

        for key, value in doc.items():
            if key == "file_path":
                file_path = value

            if key == "first_character_index":
                first = value

            if key == "last_character_index":
                last = value
        ref_list.append(f"{file_path} [{first}:{last}]")

    return ref_list
