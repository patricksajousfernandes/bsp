import os
import time
import re
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import BedrockEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# Carregar variáveis de ambiente
load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
pc = Pinecone(api_key=pinecone_api_key)

def process_docx(uploaded_file, client_name, meeting_date):
    print("Carregando DOCX...")

    # Criar um arquivo temporário
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as temp_file:
        temp_path = temp_file.name
        temp_file.write(uploaded_file.getbuffer())

    try:
        # Carregar o arquivo .docx usando Docx2txtLoader
        loader = Docx2txtLoader(temp_path)
        documents = loader.load()

        # Concatenar o page_content de todos os documentos em uma única string
        all_text = "".join([doc.page_content for doc in documents if hasattr(doc, 'page_content')])

        # Inicializar o RecursiveCharacterTextSplitter com separadores especificados
        splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=200, separators=["\n\n"], keep_separator=True)

        # Dividir o texto concatenado
        chunks = splitter.split_text(all_text)

        # Remover caracteres não alfanuméricos e converter para minúsculas
        client_name_clean = re.sub(r'\W+', '', client_name).lower()
        meeting_date_clean = re.sub(r'\W+', '-', meeting_date).lower()  # Substituir não alfanuméricos por hífens

        # Certificar que o nome do índice está no formato correto
        index_name = f"{client_name_clean}-{meeting_date_clean}"

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
        PineconeVectorStore.from_texts(chunks, embeddings, index_name=index_name)

        print("Concluído")
        return index_name

    finally:
        # Garantir que o arquivo temporário seja removido
        os.remove(temp_path)
