import os
from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

if __name__ == "__main__":
    print("Carregando DOCX...")
    # Load the .docx file using Docx2txtLoader
    loader = Docx2txtLoader("Apresentação de Proposta - BSP & Voip Do Brasil.docx")
    documents = loader.load()
    
    # Initialize the RecursiveCharacterTextSplitter with specified separators
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50,separators=["\n\n"], keep_separator=True)
    
    # Assuming that 'documents' is a list of Document objects, we need to iterate over it
    for document in documents:
        # Check if the document has the 'page_content' attribute
        if hasattr(document, 'page_content'):
            # Split the document's text using the split_text method
            chunks = splitter.split_text(document.page_content)
            # Print the resulting chunks
            print(chunks)
        else:
            print("Document does not have 'page_content' attribute.")
