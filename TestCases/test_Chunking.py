import unittest
from typing import List
from unittest.mock import patch, MagicMock

# Assuming the Chunking abstract class, AgenticChunking, and other dependencies are in chunking.py
from chunking import AgenticChunking, Chunking

class TestAgenticChunking(unittest.TestCase):
    
    def setUp(self):
        # Setup the chunker with default model configurations
        self.chunker = AgenticChunking()

    @patch('chunking.AzureChatOpenAI')  # Mock the AzureChatOpenAI class
    def test_getChunks(self, MockAzureChatOpenAI):
        # Mocking the LLM response for chunking
        mock_llm = MockAzureChatOpenAI.return_value
        mock_llm.invoke.return_value = ["chunk1", "chunk2", "chunk3"]

        text_data = "Your large document or book goes here."
        chunks = self.chunker.getChunks(text_data)

        # Verify the mock LLM was called correctly
        mock_llm.invoke.assert_called_once_with({"document": text_data})

        # Check if the returned chunks are as expected
        expected_chunks = ["chunk1", "chunk2", "chunk3"]
        self.assertEqual(chunks, expected_chunks)

    @patch('chunking.AzureChatOpenAI')
    def test_getChunks_empty_text(self, MockAzureChatOpenAI):
        # Mocking the LLM response for an empty document
        mock_llm = MockAzureChatOpenAI.return_value
        mock_llm.invoke.return_value = []

        text_data = ""  # Empty document
        chunks = self.chunker.getChunks(text_data)

        # Verify the mock LLM was called with an empty document
        mock_llm.invoke.assert_called_once_with({"document": text_data})

        # Check if the returned chunks are empty as expected
        self.assertEqual(chunks, [])

    @patch('chunking.AzureChatOpenAI')
    def test_getChunks_large_text(self, MockAzureChatOpenAI):
        # Mocking the LLM response for a large document
        mock_llm = MockAzureChatOpenAI.return_value
        mock_llm.invoke.return_value = ["chunk1", "chunk2"]

        text_data = "This is a very large document with multiple sections and paragraphs." * 100  # Large document
        chunks = self.chunker.getChunks(text_data)

        # Check that the LLM was invoked
        mock_llm.invoke.assert_called_once_with({"document": text_data})

        # Verify the result
        expected_chunks = ["chunk1", "chunk2"]
        self.assertEqual(chunks, expected_chunks)

# Run the tests
if __name__ == '__main__':
    unittest.main()
