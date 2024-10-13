from abc import ABC, abstractmethod
import os

class VectorDB(ABC):
    @abstractmethod
    def getVectorDB(self, path: str) -> object:
        pass

class Milvus_VDB(VectorDB):
    def __init__(self, directory):
        self.directory = f'data/{directory}'
        self.client = None  # Initialize client as a class attribute

    def setupVectorDB(self):
        from pymilvus import MilvusClient
        self.client = MilvusClient("context_knowledge.db", uri="http://localhost:19530", token="root:Milvus")

    def verifyCollection(self):
        if self.client.has_collection(collection_name=self.directory):
            pass
        else:
            self.client.create_collection(
                collection_name=self.directory,
                dimension=768  # Corrected typo
            )

    def getVectorDB(self):
        from langchain.indexes import VectorstoreIndexCreator
        from langchain.indexes.vectorstore import VectorStoreIndexWrapper
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
        from pymilvus import model
        import numpy as np

        embedding_fn = model.DefaultEmbeddingFunction()
        print(f'\n===== INITIATING Milvus VECTOR DB for {self.directory} framework =====')
        
        # Setup VectorDB client and verify collection
        self.setupVectorDB()
        self.verifyCollection()

        # Load documents
        print(f'Manuals dir: {self.directory}')
        loader = PyPDFDirectoryLoader(self.directory)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)
        print(f'Loaded --{self.directory}-- manuals with total {len(docs)} pages')

        # Insert embeddings
        print("Inserting vectors into vector DB...")
        vectors = embedding_fn.encode_documents(docs)  # Corrected vectors extraction
        vector_data = [
            {"id": i, "vector": vectors[i], "text": docs[i], "framework": self.directory}
            for i in range(len(vectors))
        ]

        self.client.insert(self.directory, data=vector_data)
        print('===== END OF VECTOR LOADING =====\n')

        return vector_data

    
class FAISS_VDB(VectorDB):
    def __init__(self, directory):
        """
        Initiate the object with the server or application framework e.g. dotnet, ibmbpm, linux etc
        """
        self.directory=f'data/{directory}'
        
    def getVectorDB(self):
        import numpy as np
        from langchain.vectorstores import FAISS
        from langchain.indexes import VectorstoreIndexCreator
        from langchain.indexes.vectorstore import VectorStoreIndexWrapper
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
        from langchain.embeddings import BedrockEmbeddings
        from utils import bedrock
        
        boto3_bedrock = bedrock.get_bedrock_client(
            assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
            region=os.environ.get("AWS_DEFAULT_REGION", None)
        )
        bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=boto3_bedrock)
        
        print(f'\n===== INITIATING FAISS VECTOR DB for {self.directory} framework=====')
        
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
        
        # Embedding Heartbeat check
        try:
            sample_embedding = np.array(bedrock_embeddings.embed_query(docs[0].page_content))
            print(f'Sample embedding of a document chunk: {sample_embedding}\nSize of the embedding: {sample_embedding.shape}')
        except ValueError as error:
            if  "AccessDeniedException" in str(error):
                print(f"\x1b[41m{error}\
                \nTo troubeshoot this issue please refer to the following resources.\
                 \nhttps://docs.aws.amazon.com/IAM/latest/UserGuide/troubleshoot_access-denied.html\
                 \nhttps://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html\x1b[0m\n")      
                class StopExecution(ValueError):
                    def _render_traceback_(self):
                        pass
                raise StopExecution        
            else:
                raise error
        
        # Insert embeddings
        print("Inserting vectors into vector DB...")
        vectorstore_faiss = FAISS.from_documents(
            docs,
            bedrock_embeddings,
        )
        wrapper_store_faiss = VectorStoreIndexWrapper(vectorstore=vectorstore_faiss)
        print('===== END OF VECTOR LOADING =====\n')
        
        return vectorstore_faiss  
    
class Chroma_VDB(VectorDB):
    def __init__(self, directory):
        self.directory = f'data/{directory}'

    def getVectorDB(self):
        import numpy as np
        from langchain.vectorstores import Chroma
        from langchain.indexes.vectorstore import VectorStoreIndexWrapper
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.document_loaders import PyPDFDirectoryLoader
        from langchain.embeddings import BedrockEmbeddings
        from utils import bedrock

        boto3_bedrock = bedrock.get_bedrock_client(
            assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
            region=os.environ.get("AWS_DEFAULT_REGION", None)
        )
        bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-image-v1", client=boto3_bedrock)

        print(f"\n====== INITIATING CHROMA VECTOR DB for {self.directory} framework======")

        loader = PyPDFDirectoryLoader(self.directory)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
        )
        docs = text_splitter.split_documents(documents)

        print(f'Loaded --{self.directory}-- manuals with total {len(docs)} pages')
        
        print("Inserting vectors into vector DB...")
        vectorstore_chroma = Chroma.from_documents(docs, bedrock_embeddings)
        wrapper_store_chroma = VectorStoreIndexWrapper(vectorstore=vectorstore_chroma)
        print('===== END OF VECTOR LOADING =====\n')

        return vectorstore_chroma

"""
        try:
            # Ensure sample_text is a string
            sample_text = docs[0].page_content
            if isinstance(sample_text, list):
                sample_text = " ".join(sample_text)  # Join list into a single string
            else:
                sample_text = str(sample_text)
                
            formatted_sample_text = sample_text.replace(os.linesep, " ")

            sample_embedding = np.array(bedrock_embeddings.embed_query([formatted_sample_text]))

            print(f'Sample embedding of a document chunk: {sample_embedding}\nSize of the embedding: {sample_embedding.shape}')
        except ValueError as error:
            if "AccessDeniedException" in str(error):
                print(f"\x1b[41m{error}\nTo troubleshoot this issue please refer to the following resources.\nhttps://docs.aws.amazon.com/IAM/latest/UserGuide/troubleshoot_access-denied.html\nhttps://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html\x1b[0m\n")
                class StopExecution(ValueError):
                    def _render_traceback_(self):
                        pass
                raise StopExecution
            else:
                raise error
"""

class Milvus_VDB(VectorDB):
    def __init__(self, directory):
        """
        Initiate the object with the server or application framework e.g. dotnet, ibmbpm, linux etc
        """
        self.directory=f'data/{directory}'
        
    def getVectorDB(self):
        import numpy as np
        from langchain.vectorstores import FAISS
        from langchain.indexes import VectorstoreIndexCreator
        from langchain.indexes.vectorstore import VectorStoreIndexWrapper
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
        from langchain.embeddings import BedrockEmbeddings
        from utils import bedrock
        
        boto3_bedrock = bedrock.get_bedrock_client(
            assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
            region=os.environ.get("AWS_DEFAULT_REGION", None)
        )
        bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=boto3_bedrock)
        
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
        
        # Embedding Heartbeat check
        try:
            sample_embedding = np.array(bedrock_embeddings.embed_query(docs[0].page_content))
            print(f'Sample embedding of a document chunk: {sample_embedding}\nSize of the embedding: {sample_embedding.shape}')
        except ValueError as error:
            if  "AccessDeniedException" in str(error):
                print(f"\x1b[41m{error}\
                \nTo troubeshoot this issue please refer to the following resources.\
                 \nhttps://docs.aws.amazon.com/IAM/latest/UserGuide/troubleshoot_access-denied.html\
                 \nhttps://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html\x1b[0m\n")      
                class StopExecution(ValueError):
                    def _render_traceback_(self):
                        pass
                raise StopExecution        
            else:
                raise error
        
        # Insert embeddings
        print("Inserting vectors into vector DB...")
        vectorstore_Milvus = Milvus.from_documents(
            docs,
            bedrock_embeddings,
        )
        wrapper_store_Milvus = VectorStoreIndexWrapper(vectorstore=vectorstore_faiss)
        print('===== END OF VECTOR LOADING =====\n')
        
        return vectorstore_Milvus