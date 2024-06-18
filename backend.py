import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_aws import BedrockLLM
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import BedrockEmbeddings
from langchain.memory import ConversationBufferMemory

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Inicializa as embeddings e o LLM com parâmetros especificados
embeddings = BedrockEmbeddings(
    credentials_profile_name='default',
    model_id='cohere.embed-multilingual-v3'
)

llm = BedrockLLM(
    credentials_profile_name='default',
    model_id='anthropic.claude-v2',
    model_kwargs={
        "max_tokens_to_sample": 3000,
        "temperature": 0.0,
        "top_p": 0.9
    }
)

# Inicializa o Pinecone Vector Store com o nome do índice das variáveis de ambiente
vectorstore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"], embedding=embeddings
)

# Recupera o prompt de chat de QA de recuperação do hub
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

# Função para formatar a resposta da LLM
def formatar_resposta_llm(resposta):
    # Extrai o conteúdo relevante da resposta da LLM
    conteudo_resposta = resposta['answer'] if 'answer' in resposta else resposta
    
    # Divide a resposta em linhas
    linhas_resposta = conteudo_resposta.split('\n')
    
    # Formata cada linha como um tópico claro e legível
    resposta_formatada = "\n".join([f"- **{linha.strip()}**" for linha in linhas_resposta if linha.strip()])
    
    return resposta_formatada

# Função para utilizar o LLM com Pinecone e RAG
def utilizar_llm_com_pinecone_e_RAG(input_text, conversation_history):
    print("Iniciando a recuperação com RAG e memória de conversa...")

    # Inicializa a memória do chatbot com o histórico da conversa
    memory = ConversationBufferMemory(conversation_history=conversation_history)

    # Define a consulta para o LLM
    query = input_text
    
    # Cria uma cadeia para processar a consulta com o LLM
    chain = PromptTemplate.from_template(template=query) | llm

    # Cria cadeias para combinar documentos e recuperação com memória de conversa
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    )

    # Invoca a cadeia de recuperação com a consulta e o histórico da conversa
    result = retrieval_chain.invoke(input={"input": query, "conversation_history": conversation_history})

    # Formata e exibe a resposta da LLM
    resposta_formatada = formatar_resposta_llm(result)
    
    # Atualiza o histórico da conversa com a resposta atual
    conversation_history.append({"input": query, "response": resposta_formatada})

    # Retorna a resposta formatada e a memória atualizada
    return resposta_formatada, conversation_history