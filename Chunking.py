import os, nltk
import numpy as np
from abc import ABC, abstractmethod
from typing import List
from flair.models import SequenceTagger
from flair.data import Sentence
from flair.splitter import SegtokSentenceSplitter
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv, find_dotenv
from tqdm import tqdm
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate

class Chunking(ABC):
    @abstractmethod
    def getChunks(self, file_name: str) -> list[str]:
        """
        Chunks the file given in the input/file_name by certain amount of line. A particular chunk may contain a few lines from the file.
        file_name : str
        
        return an list of chunks.
        """
        pass

class LineBasedChunk(Chunking):
    def __init__(self, file_name):
        self.path=f'input/{file_name}'
        
    def getChunks(self):
        print(f'\n===== CHUNKING LOG FILE =====\n')
        lines_per_chunk = 10
        all_chunks = []

        with open(self.path) as bigfile:
            chunk=""
            for lineno, line in enumerate(bigfile):
                if (lineno+1) % lines_per_chunk == 0:
                    all_chunks.append(chunk)
                    chunk=""
                chunk += '\n' + line

            all_chunks.append(chunk)

        print(f'Total {len(all_chunks)} chunks extracted.\n===== END OF LOG CHUNKING =====\n')
        return all_chunks  
    
class FixedWindowOverlapChunking(Chunking):
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 10):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def getchunks(self, file_name: str) -> List[str]:
        """Splits text at the given chunk_size, and starts the next chunk from start - chunk_overlap position"""
        with open(f'input/{file_name}','r') as file:
            text = file.read()

        all_chunks = []
        start = 0
        
        while start <= len(text):
            end = start + self.chunk_size
            all_chunks.append(text[start:end])
            start = end - self.chunk_overlap
        
        print(f'\n===== CHUNKING FILE USING FIXED WINDOW AND OVERLAP=====\n')
        print(f'Total {len(chunks)} chunks extracted.\n===== END OF CHUNKING =====\n')
        return all_chunks

class SemanticChunking(Chunking):
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 10):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def getchunks(self, file_name: str) -> List[str]:
        with open(f'input/{file_name}','r') as file:
            text = file.read()

        splitter = SegtokSentenceSplitter()

        #Split text into sentences
        sentences = splitter.Split(text)

        all_chunks = []
        current_chunk = ""

        for sentence in sentences:
            #Add sentence to the current chunk
            if len(current_chunk) + len(sentence.to_plain_string()) <= chunk_size:
                current_chunk += " " + sentence.to_plain_string()
            else:
                #If adding the next sentence exceeds max size, start a new chunk
                all_chunks.append(current_chunk.strip())
                current_chunk = sentence.to_plain_string()

        # Add the last chunk if it exists
        if current_chunk:
            all_chunks.append(current_chunk.strip())

        return all_chunks

class EmbeddingChunking(Chunking):
    def __init__(self, chunk_size: int = 400):
        self.chunk_size = chunk_size

    def embedding_splitter(self, file_name: str) -> List[str]:
        with open(f'input/{file_name}','r') as file:
            text = file.read()

        load_dotenv(find_dotenv())

        """
        #Set Azure OpenAI API environment variables (ensure these are set in your environment)
        #You can also set these in your environment directly
        os.environ["OPENAI_API_KEY"] = "your-azure-openai-api-key"
        os.environ["OPENAI_API_BASE"] = "your-azure-openai-api-endpoint"
        os.environ["OPENAI_API_VERSION"] = "2023-05-15"
    
        #Initialize OpenAIEmbeddings using LangChain's Azure support
        embedding_model = AzureOpenAIEmbeddings(deployment="text-embedding-ada-002-01")  # Use your Azure model name
        """

        #Step 1: Split the text into sentences
        def split_into_sentences(text):
            splitter = SegtokSentenceSplitter()

            # Split text into sentences
            sentences = splitter.split(text)
            Sentence_str = []

            for sentence in sentences:
                Sentence_str.append(sentence.to_plain_string())
            return Sentence_str[:100]
    
        #step 2: get embeddings for each sentence
        def get_embeddings(sentences):
            embeddings = []
            for sentence in tqdm(sentences, desc="Generating embeddings"):
                embedding = embedding_model.embed_documents([sentence]) #embeds a single sentence
                embeddings.append(embedding[0]) #embed_documents returns a list, so take the first element
            return embeddings

        # step3: from chunks based on sentence embeddings, a similarity threshold, and a max chunk character size
        def form_chunks(sentences, embeddings, similarity_threshold=0.7, chunk_size=500):
            all_chunks = []
            current_chunk = []
            current_chunk_emb = []
            current_chunk_length = 0 #track the character length of the current chunk

            for i, (sentence, emb) in enumerate(zip(sentences, embeddings)):
                emb = np.arry(emb) #ensure the embedding is a numpy array
                sentence_length = len(sentence) #calculate the length of the sentence

                if current_chunk:
                    #calculate similarity with the current chunk's embedding (mean of the embeddings in the chunk)
                    chunk_emb = np.mean(np.array(current_chunk_emb), axis=0).reshape(1, -1) #average embedding of the chunk
                    similarity = cosine_similarity(emb.reshape(1, -1), chunk_emb)[0][0]

                    if similarity < similarity_threshold or current_chunk_length + sentence_length > chunk_size:
                        # If similarity is below threshold or adding this sentence exceeds max chunk size, create a new chunk
                        all_chunks.append(current_chunk)
                        current_chunk = [sentence]
                        current_chunk_emb = [emb]
                        current_chunk_length = sentence_length  # Reset chunk length
                    else:
                        # Else, add sentence to the current chunk
                        current_chunk.append(sentence)
                        current_chunk_emb.append(emb)
                        current_chunk_length += sentence_length  # Update chunk length
                else:
                    current_chunk.append(sentence)
                    current_chunk_emb = [emb]
                    current_chunk_length = sentence_length  # Set initial chunk length
            # Add the last chunk
            if current_chunk:
                all_chunks.append(current_chunk)
            return all_chunks

        # Apply the sentence splitting
        sentences = split_into_sentences(text_data)
    
        # Get sentence embeddings
        embeddings = get_embeddings(sentences)
    
        # Form chunks based on embeddings
        chunks = form_chunks(sentences, embeddings, chunk_size=chunk_size)
    
        return chunks

class AgenticChunking(Chunking):
    def __init__(self, model: str = "gpt-4o", api_version: str = "2023-03-15-preview", temperature: int = 1):
        self.llm = AzureChatOpenAI(
            model = model,
            api_version = api_version,
            verbose = True,
            temperature = temperature
        )

    self.prompt_template = PromptTemplate.from_template(
       """I am providing a document below. 
            Please split the document into chunks that maintain semantic coherence and ensure that each chunk represents a complete and meaningful unit of information. 
            Each chunk should stand alone, preserving the context and meaning without splitting key ideas across chunks. 
            Use your understanding of the content’s structure, topics, and flow to identify natural breakpoints in the text. 
            Ensure that no chunk exceeds 1000 characters length, and prioritize keeping related concepts or sections together.

            Do not modify the document, just split to chunks and return them as an array of strings, where each string is one chunk of the document.
            Return the entire book without stopping between any sentences.

            Document:
            {document}
            """ 
    )

    def getChunks(self, file_name: str) -> List[str]:
        """Splits text_data into semantically coherent chunks using an LLM-based approach."""
        chain = self.prompt_template | self.llm
        result = chain.invoke({"document": text_data})
        return result