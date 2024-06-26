from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain import hub
from langchain_aws import ChatBedrock
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import BedrockEmbeddings
from pinecone.exceptions import NotFoundException as PineconeNotFoundException
import re
import os

def get_index_name(input_text):
    # Carregar variáveis de ambiente
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    index_name1 = os.getenv("INDEX_NAME1")

    embeddings = BedrockEmbeddings(
        api_key=pinecone_api_key,
        credentials_profile_name="default",
        region_name="us-east-1",
        model_id="amazon.titan-embed-text-v2:0",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

    llm_model = ChatBedrock(
        credentials_profile_name='default',
        region_name="us-east-1",
        model_id='anthropic.claude-3-5-sonnet-20240620-v1:0',
        model_kwargs={
            "max_tokens": 3000,
            "temperature": 0.1,
            "top_p": 0.9
        },
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

    match = re.search(r"Cliente:\s*(.*?)\s*Data:\s*(\d{2}/\d{2}/\d{4})", input_text)
    if match:
        client_name = match.group(1)
        meeting_date = match.group(2)
        
        client_name_clean = re.sub(r'\W+', '', client_name).lower()
        meeting_date_clean = re.sub(r'\W+', '-', meeting_date).lower()
        index_name = f"{client_name_clean}-{meeting_date_clean}"
        
        return process_input(input_text, index_name, embeddings, llm_model, index_name1)
    else:
        return process_input(input_text, index_name1, embeddings, llm_model, index_name1)

def process_input(input_text, index_name, embeddings, llm_model, default_index_name):
    try:
        vectorstore = PineconeVectorStore(
            index_name=index_name, embedding=embeddings
        )
    except PineconeNotFoundException:
        print(f"Índice '{index_name}' não encontrado no Pinecone.")
        return None
    
    retriever = vectorstore.as_retriever()
    
    if index_name == default_index_name:
        retrieval_qa_chat_prompt = hub.pull("texte/awsprompt")
    else:
        retrieval_qa_chat_prompt = hub.pull("texte/resumidordetexto")
    
    combine_docs_chain = create_stuff_documents_chain(llm_model, retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    
    result = retrieval_chain.invoke({"input": input_text})
    
    return result['answer']
