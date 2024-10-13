from abc import ABC, abstractmethod
import os

class LLM(ABC):
    @abstractmethod
    def get(self, typ) -> object:
        """
        Get the LLM object.
        
        typ: 'light' or 'heavy'
        return an object
        """
        pass
    
class Meta(LLM):
    def get(self, typ='light'):
        from utils import bedrock
        from langchain.llms.bedrock import Bedrock
        boto3_bedrock = bedrock.get_bedrock_client(
            assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
            region=os.environ.get("AWS_DEFAULT_REGION", None)
        )
        
        if typ not in ['light', 'heavy']:
            raise ValueError(f"Invalid type. Expected one of: {'light', 'heavy'}")
            
        if typ == 'light':
            the_model='meta.llama3-8b-instruct-v1:0'
            print(f'\n===== INITIATING LLM: {the_model} =====')
            return Bedrock(model_id=the_model, client=boto3_bedrock, model_kwargs={'max_tokens_to_sample':8000, 'temperature': 0.3})  
        else:
            the_model='meta.llama3-70b-instruct-v1:0'
            print(f'\n===== INITIATING LLM: {the_model} =====')
            return Bedrock(model_id=the_model, client=boto3_bedrock, model_kwargs={'max_tokens_to_sample':8000, 'temperature': 0.3})
        
    
class Titan(LLM):
    def get(self, typ='light'):
        from utils import bedrock
        from langchain.llms.bedrock import Bedrock
        boto3_bedrock = bedrock.get_bedrock_client(
            assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
            region=os.environ.get("AWS_DEFAULT_REGION", None)
        )
        
        if typ not in ['light', 'heavy']:
            raise ValueError(f"Invalid type. Expected one of: {'light', 'heavy'}")
            
        if typ == 'light':
            the_model='amazon.titan-text-lite-v1'
            print(f'\n===== INITIATING LLM: {the_model} =====')
            return Bedrock(model_id=the_model, client=boto3_bedrock, model_kwargs={'max_tokens_to_sample':4000})  
        else:
            the_model='amazon.titan-text-express-v1'
            print(f'\n===== INITIATING LLM: {the_model} =====')
            return Bedrock(model_id=the_model, client=boto3_bedrock, model_kwargs={'max_tokens_to_sample':8000})
            
            
class Mistral(LLM): #https://aws.amazon.com/bedrock/mistral/; https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html
    def get(self, type='light', max_token=500):
        from utils import bedrock
        from langchain.llms.bedrock import Bedrock
        boto3_bedrock = bedrock.get_bedrock_client(
            assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
            region=os.environ.get("AWS_DEFAULT_REGION", None)
        )

        if type not in ['light', 'heavy']:
            raise ValueError(f"Invalid type. Expected one of: {'light', 'heavy'}")

        if type == 'light':
            the_model='mistral.mistral-7b-instruct-v0:2'
            print(f'\n===== INTIATING LLM: {the_model} =====')
            return Bedrock(model_id=the_model, client=boto3_bedrock, model_kwargs={'max_token_to_sample':32000})
        else:
            the_model='mistral.mistral-large-2402-v1:0'
            print(f'\n===== INTIATING LLM: {the_model} =====')
            return Bedrock(model_id=the_model, client=boto3_bedrock, model_kwargs={'max_token_to_sample':32000})

""" Anthropic is available only for enterprise or company so ruled out

class Anthropic(LLM):
    def get(self, typ='light', max_token=500):
        from utils import bedrock
        from langchain.llms.bedrock import Bedrock
        boto3_bedrock = bedrock.get_bedrock_client(
            assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
            region=os.environ.get("AWS_DEFAULT_REGION", None)
        )
        
        if typ not in ['light', 'heavy']:
            raise ValueError(f"Invalid type. Expected one of: {'light', 'heavy'}")
            
        if typ == 'light':
            the_model='anthropic.claude-instant-v1'
            print(f'\n===== INITIATING LLM: {the_model} =====')
            return Bedrock(model_id=the_model, client=boto3_bedrock, model_kwargs={'max_tokens_to_sample':max_token, 'temperature': 0.3})  
        else:
            the_model='anthropic.claude-v2'
            print(f'\n===== INITIATING LLM: {the_model} =====')
            return Bedrock(model_id=the_model, client=boto3_bedrock, model_kwargs={'max_tokens_to_sample':max_token, 'temperature': 0.3})   

 Cohere LLM is not available in AWS Bedrock, we have only Cohere embeddings

class Cohere(LLM): #https://docs.cohere.com/docs/models
    def get(self, type='light', max_token=500):
        from utils import bedrock
        from langchain.llms.bedrock import Bedrock
        boto3_bedrock = bedrock.get_bedrock_client(
            assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
            region=os.environ.get("AWS_DEFAULT_REGION", None)
        )

        if type not in ['light', 'heavy']:
            raise ValueError(f"Invalid type. Expected one of: {'light', 'heavy'}")

        if type == 'light':
            the_model='cohere.command-light-text-v14'
            print(f'\n===== INTIATING LLM: {the_model} =====')
            return Bedrock(model_id=the_model, client=boto3_bedrock, model_kwargs={'max_token_to_sample':200})
        else:
            the_model='cohere.command-text-v14'
            print(f'\n===== INTIATING LLM: {the_model} =====')
            return Bedrock(model_id=the_model, client=boto3_bedrock, model_kwargs={'max_token_to_sample':200})
"""