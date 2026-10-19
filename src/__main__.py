import sys
import fire

from src.cli.cli import CLInterface


end = "\033[0m"
r = "\033[31m\033[5m\033[1m"
g = "\033[32m\033[5m\033[1m"


def main() -> None:
    try:
        fire.Fire(CLInterface)

    except Exception as e:
        print(f"{r}[ERROR]{end}: An unexpected error occured:\n-> {e}")



if __name__ == "__main__":
    try:
        sys.exit(main())

    except KeyboardInterrupt:
        print(f"\n{g}[INFO]{end}: Quitting RAG!")

    except Exception as e:
        print(f"{r}[ERROR]{end}: An unexpected error occured:\n-> {e}")
