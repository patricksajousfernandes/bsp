import os
from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import BedrockEmbeddings
from langchain_pinecone import PineconeVectorStore

# Load environment variables
load_dotenv()

print("Carregando DOCX...")
# Load the .docx file using Docx2txtLoader
loader = Docx2txtLoader("Apresentação de Proposta - BSP Cloud  & Now Seguros.docx")
documents = loader.load()

# Concatenate the page_content of all documents into a single string
all_text = "".join([doc.page_content for doc in documents if hasattr(doc, 'page_content')])

# Initialize the RecursiveCharacterTextSplitter with specified separators
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50,separators=["\n\n"], keep_separator=True)

# Split the concatenated text
chunks = splitter.split_text(all_text)

# Initialize the BedrockEmbeddings object
embeddings = BedrockEmbeddings(
    credentials_profile_name="default",
    region_name="us-east-1",
    model_id="amazon.titan-embed-text-v1"
)

# Create metadata for each chunk


# Add texts and metadata to the Pinecone index
PineconeVectorStore.from_texts(chunks, embeddings, index_name=os.environ['INDEX_NAME'])
