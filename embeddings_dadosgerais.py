import os
import time
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_aws import BedrockEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# Carregar variáveis de ambiente
load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
pc = Pinecone(api_key=pinecone_api_key)

def process_pdf(uploaded_file):
    print("Carregando PDF...")

    # Criar um arquivo temporário
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_path = temp_file.name
        temp_file.write(uploaded_file.getbuffer())

    try:
        # Carregar o arquivo PDF usando PyPDFLoader
        loader = PyPDFLoader(temp_path)
        pages = loader.load()

        # Inicializar o CharacterTextSplitter com separadores especificados
        text_splitter = CharacterTextSplitter(
            separator=r'\n\n',
            chunk_size=1000,
            chunk_overlap=200,
        )

        # Dividir o texto em chunks
        chunks = text_splitter.split_documents(pages)

        # Certificar que o nome do índice está no formato correto
        index_name = "geral"  # Nome fixo para o índice

        # Inicializar o objeto BedrockEmbeddings
        embeddings = BedrockEmbeddings(
            region_name="us-east-1",
            model_id="amazon.titan-embed-text-v2:0",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )

        # Verificar se o índice já existe
        existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

        if index_name not in existing_indexes:
            pc.create_index(
                name=index_name,
                dimension=1024,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            while not pc.describe_index(index_name).status["ready"]:
                time.sleep(1)

        index = pc.Index(index_name)

        # Adicionar textos e metadados ao índice Pinecone
        PineconeVectorStore.from_documents(pages, embeddings, index_name=index_name)

        print("Concluído")
        return index_name

    finally:
        # Garantir que o arquivo temporário seja removido
        os.remove(temp_path)
