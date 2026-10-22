from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

from src.utils.file_utils import check_file_extension
from src.models.models import MinimalSource

CHUNK_OVERLAP_MULT = 0.15
EXTENSIONS = [".py", ".md", ".txt"]


class Chunker():
    """
    Splits the source files to create chunks and preserves the context.

    Attributes:
      - create_chunks(self, file: str, content: str) -> MinimalSource | None
      - _chunk_py_file(self, file: str, content: str) -> MinimalSource
      - _chunk_text_file(self, extension: str,
                         file: str, content: str) -> MinimalSource
      - _file_split(self, splitter: RecursiveCharacterTextSplitter,
                    file_path: str, content: str) -> MinimalSource
    """
    def __init__(self, chunk_size: int) -> None:
        """
        Initialize the chunking.

        Parameter:
          - chunk_size: int
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = self.chunk_size * CHUNK_OVERLAP_MULT

    def create_chunks(self, file: str, content: str) -> MinimalSource | None:
        """
        Creates the chunks according the file extension.

        Parameter:
          - file: str
          - content: str

        Return
        -> MinimalSource | None
        """
        extension = check_file_extension(file)

        if extension == ".py":
            return self._chunk_py_file(file, content)

        elif extension == ".md" or extension == ".txt":
            return self._chunk_text_file(extension, file, content)

        else:
            print(f"'{file}' doesn't have the expected extension. "
                  f"({extension} is not part of the desired extensions)")
        return None

    # --------------- #
    # Private Methods #
    # --------------- #
    def _chunk_py_file(self, file: str, content: str) -> MinimalSource:
        """
        Chunks python code.

        Parameter:
          - file: str

        Return
        -> MinimalSource
        """
        python_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        return self._file_split(python_splitter, file, content)

    def _chunk_text_file(self, extension: str,
                         file: str, content: str) -> MinimalSource:
        """
        Chunks Markdown and text files.

        Parameters:
          - extension: str
          - file: str
          - content: str

        Return
        -> MinimalSource
        """
        if extension == ".md":
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.MARKDOWN,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )

        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                keep_separator=True
            )

        return self._file_split(splitter, file, content)

    def _file_split(self, splitter: RecursiveCharacterTextSplitter,
                    file_path: str, content: str) -> MinimalSource:
        """
        Splits the source file in the correct format.

        Parameters:
          - splitter: RecursiveCharacterTextSplitter
          - file_path: str
          - content: str

        Return
        -> MinimalSource
        """
        pass
