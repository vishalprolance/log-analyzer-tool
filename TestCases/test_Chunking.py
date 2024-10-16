import string

class RecursiveCharacterTextSplitter:
    """Splits text into chunks using a recursive approach based on character separators.

    Args:
        chunk_size (int, optional): The maximum size of each chunk in characters. Defaults to 1000.
        chunk_overlap (int, optional): The number of characters to overlap between chunks. Defaults to 10.
        separators (list[str], optional): A list of characters to use as separators. Defaults to a list of common punctuation marks and whitespace characters.
    """

    def __init__(self, chunk_size=1000, chunk_overlap=10, separators=None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        if separators is None:
            self.separators = list(string.punctuation) + [" "]
        else:
            self.separators = separators

    def split_text(self, text):
        """Splits the given text into chunks.

        Args:
            text (str): The text to split.

        Returns:
            list[str]: A list of chunks.
        """

        chunks = []
        start = 0

        while start <= len(text):
            end = start + self.chunk_size
            if end > len(text):
                end = len(text)

            chunk = text[start:end]

            if len(chunk) > self.chunk_size:
                # If the chunk is too long, try splitting it using the separators
                for separator in self.separators:
                    if separator in chunk:
                        split_index = chunk.rindex(separator)
                        subchunk = chunk[:split_index + 1]
                        if len(subchunk) <= self.chunk_size:
                            chunks.append(subchunk)
                            start = start + split_index + 1
                            break
                else:
                    # If no separator was found, just take the entire chunk
                    chunks.append(chunk)
                    start = end
            else:
                chunks.append(chunk)
                start = end - self.chunk_overlap

        return chunks