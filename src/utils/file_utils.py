import os

from pathlib import Path

EXTENSIONS = [".py", ".md", ".txt"]


def check_file_extension(file: str) -> str | None:
    """
    Checks if the file has one of the wanted extensions and returns it.

    Parameter:
      - file: str

    Return
    -> str
    """
    file_path = Path(file)

    for extension in EXTENSIONS:
        if file_path.suffix == extension:
            return extension

    return None


def load_files(vLLM_path: str) -> dict[str, str]:
    """
    Creates a list with every file title and content.

    Parameter:
      - vLLM_path: str

    Return
    -> dict[str, str]
    """
    files_dict: dict[str, str] = {}

    # Checks every file in the vLLM path
    for root, dirs, files in os.walk(vLLM_path, topdown=True,
                                     onerror=None, followlinks=False):

        for file in files:
            # Checks the file's extension
            for extension in EXTENSIONS:
                if not check_file_extension(file):
                    continue

                # Assemble the entire file path
                file_path: str = os.path.join(root, file)

                # Open the file to extract its content
                try:
                    with open(file_path) as f:
                        files_dict[file_path] = f.read()

                except PermissionError:
                    raise ValueError("Permission error with this file: "
                                     f"{file}")

                except Exception as e:
                    raise ValueError(e)

    return files_dict
