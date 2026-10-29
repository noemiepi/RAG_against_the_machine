from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

from src.utils.file_utils import check_file_extension
from src.models.models import MinimalSource

CHUNK_OVERLAP_MULT = 0.15
EXTENSIONS = [".py", ".md", ".txt"]


class Chunker():
    """
    Splits the source files to create chunks and preserves the context.

    Attributes:
      - create_chunks(self, file: str,
                      content: str) -> list[MinimalSource] | None
      - _chunk_py_file(self, file: str, content: str) -> list[MinimalSource]
      - _chunk_text_file(self, extension: str, file: str,
                         content: str) -> list[MinimalSource]
      - _file_split(self, splitter: RecursiveCharacterTextSplitter,
                    file_path: str, content: str) -> list[MinimalSource]
    """
    def __init__(self, chunk_size: int) -> None:
        """
        Initialize the chunking.

        Parameter:
          - chunk_size: int
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = self.chunk_size * CHUNK_OVERLAP_MULT

    def create_chunks(self, file: str,
                      content: str) -> list[MinimalSource] | None:
        """
        Creates the chunks according the file extension.

        Parameters:
          - file: str
          - content: str

        Return
        -> list[MinimalSource] | None
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
    def _chunk_py_file(self, file: str, content: str) -> list[MinimalSource]:
        """
        Chunks python code.

        Parameters:
          - file: str
          - content: str

        Return
        -> list[MinimalSource]
        """
        python_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        return self._file_split(python_splitter, file, content)

    def _chunk_text_file(self, extension: str, file: str,
                         content: str) -> list[MinimalSource]:
        """
        Chunks Markdown and text files.

        Parameters:
          - extension: str
          - file: str
          - content: str

        Return
        -> list[MinimalSource]
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
                    file_path: str, content: str) -> list[MinimalSource]:
        """
        Splits the source file in the correct format and make it
        respect MinimalSource class format (file path, first and
        last character index).

        Parameters:
          - splitter: RecursiveCharacterTextSplitter
          - file_path: str
          - content: str

        Return
        -> list[MinimalSource]
        """
        chunk_list: list[MinimalSource] = []
        i: int = 0

        # Goes through every chunk of the splitter
        for chunk in splitter.split_text(content):
            first_i: int = content.find(chunk, i)

            # Search from the beginning if there was an error previously
            if first_i == -1:
                first_i = content.find(chunk)
            # It returned an error
            if first_i == -1:
                continue

            # Get the last character's index
            last_i: int = first_i + len(chunk)

            # Update the index
            i = first_i + 1

            chunk_list.append(MinimalSource(file_path=file_path,
                                            first_character_index=first_i,
                                            last_character_index=last_i))

        return chunk_list
