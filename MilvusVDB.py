from abc import ABC, abstractmethod
import os

class VectorDB(ABC):
    @abstractmethod
    def getVectorDB(self, path: str) -> object:
        pass

    
class Milvus_VDB(VectorDB):
    def __init__(self, directory):
        self.directory = f'data/{directory}'
        
    def setupVectorDB(self):
        from pymilvus import MilvusClient
        client = MilvusClient("context_knowledge.db")
    
    def verifyCollection(self):
        if client.has_collection(collection_name={self.directory}):
            pass
        else:
            client.create_collection(
                collection_name={self.directory},
                dimernsion=768, # The vectors we will use in this demo has 768 dimensions
            )

    def getVectorDB(self):
        from langchain.indexes import VectorstoreIndexCreator
        from langchain.indexes.vectorstore import VectorStoreIndexWrapper
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
        from pymilvus import model
        import numpy as np
        embedding_fn = model.DefaultEmbeddingFunction()

        print(f'\n===== INITIATING Milvus VECTOR DB for {self.directory} framework=====')
        # Load all the Manuals
        print(f'Manuals dir: {self.directory}')
        loader = PyPDFDirectoryLoader(self.directory)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap  = 100,
        )
        
        docs = text_splitter.split_documents(documents)

        print(f'Loaded --{self.directory}-- manuals with total {len(docs)} pages')
        # Insert embeddings
        print("Inserting vectors into vector DB...")
        vectorstore_milvus = embedding_fn.encode_documents(docs)
        wrapper_store_milvus = VectorStoreIndexWrapper(vectorstore=vectorstore_milvus)
        print('===== END OF VECTOR LOADING =====\n')

        return vectorstore_milvus