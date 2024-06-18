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
    credentials_profile_name="default",
    region_name="us-east-1",
    model_id="amazon.titan-embed-text-v1"
)


llm = BedrockLLM(
    credentials_profile_name='default',
    region_name="us-east-1",
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


# Função para utilizar o LLM com Pinecone e RAG
def utilizar_llm_com_pinecone_e_RAG(input_text, conversation_history):
    print("Iniciando a recuperação com RAG e memória de conversa...")

    # Inicializa a memória do chatbot com o histórico da conversa
    memory = ConversationBufferMemory(conversation_history=conversation_history)

    # Define a consulta para o LLM
    query = input_text

    prompt = """
Given the question {query}, your task is to extract the most precise and relevant information from the meeting transcriptions. If the requested information is not available in the transcriptions, clearly indicate that the information cannot be found. Avoid making assumptions or inferences that are not directly supported by the transcriptions. If a transcription is in another language, translate it to Brazilian Portuguese before extracting the information. The questions may cover a variety of topics, including but not limited to, details about the meetings available for consultation, the client's name of a meeting, the date of a meeting, the main topics of a meeting, and what the client wants in a meeting. Your response should be precise, complete, and straight to the point. Always thank at the end for having asked.
"""

    
    # Cria uma cadeia para processar a consulta com o LLM
    chain = PromptTemplate(input_variables=["query"],template=summary_template) | llm

    # Cria cadeias para combinar documentos e recuperação com memória de conversa
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    )

    # Invoca a cadeia de recuperação com a consulta e o histórico da conversa
    result = retrieval_chain.invoke(input={"input": query, "conversation_history": conversation_history})

    # Aqui você deve formatar a resposta do LLM de acordo com suas necessidades
    resposta_formatada = formatar_resposta_llm(result)  # substitua isso pela sua função de formatação

    # Atualiza o histórico da conversa com a resposta atual
    conversation_history.append({"input": query, "response": resposta_formatada})

    # Retorna a resposta formatada e a memória atualizada
    return resposta_formatada, conversation_history

# Função para formatar a resposta do LLM
def formatar_resposta_llm(resposta):
    # Aqui você deve implementar a lógica para formatar a resposta do LLM
    # Por exemplo, você pode querer extrair certos campos da resposta
    return resposta['answer']  # substitua isso pela sua função de formatação

# Inicializa o histórico da conversa como uma lista vazia
conversation_history = []

while True:
    # Recebe a entrada do usuário
    input_text = input("Digite sua pergunta (ou 'exit' para sair): ")

    # Verifica se o usuário digitou 'exit'
    if input_text.lower() == 'exit':
        break

    # Chama a função com a consulta de entrada e o histórico da conversa
    resposta, conversation_history = utilizar_llm_com_pinecone_e_RAG(input_text, conversation_history)

    # Imprime a resposta
    print("Resposta: ", resposta)
