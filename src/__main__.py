import sys
import fire

from src.rag import RAGEngine


end = "\033[0m"
r = "\033[31m\033[5m\033[1m"
g = "\033[32m\033[5m\033[1m"


def main() -> int:
    try:
        fire.Fire(RAGEngine)
        return 0

    except Exception as e:
        print(f"{r}[ERROR]{end}: An unexpected error occured:\n-> {e}")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())

    except KeyboardInterrupt:
        print(f"\n{g}[INFO]{end}: Quitting RAG!")

    except Exception as e:
        print(f"{r}[ERROR]{end}: An unexpected error occured:\n-> {e}")
