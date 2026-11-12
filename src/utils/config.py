class RAGConfig():
    """
    A class reuniting every base information for the RAGEngine class.
    """
    MAX_CHUNK = 2000
    MAX_K = 1000

    QUERY = "How to configure the OpenAI server?"

    DATASET_UNANSWERED_PATH = "data/datasets/UnansweredQuestions/"
    "dataset_docs_public.json"
    DATASET_ANSWERED_PATH = "data/datasets/AnsweredQuestions/"
    "dataset_docs_public.json"
    RESULTS_PATH = "data/output/search_results/UnansweredQuestions/"
    "dataset_docs_public.json"

    SAVE_PATH = "data/output/search_results/UnansweredQuestions"
    SEARCH_SAVE_PATH = "data/output/search_results_and_answer/"
    "UnansweredQuestions"

