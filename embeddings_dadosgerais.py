from langchain_aws import BedrockEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
import os

# Carregar o PDF
loader = PyPDFLoader("bedrock-ug.pdf")
pages = loader.load()

text_splitter = CharacterTextSplitter(
    separator=r'\\n\\n',
    chunk_size=1000,
    chunk_overlap=200,
)

texts = text_splitter.split_documents(pages)


embeddings = BedrockEmbeddings(
    credentials_profile_name="default",
    region_name="us-east-1",
    model_id="amazon.titan-embed-text-v2:0"
)


PineconeVectorStore.from_documents(pages, embeddings, index_name="geral")


print("concluido")