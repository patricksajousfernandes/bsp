import streamlit as st
from embedding import process_docx  # Certifique-se de ajustar o nome do arquivo de acordo com onde você salvar a função
from datetime import datetime
from backend2 import get_index_name

# Título do aplicativo
st.title("Chatbot BSP Cloud")

# Inicialize o histórico de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exiba as mensagens de chat do histórico na reexecução do aplicativo
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Aceite a entrada do usuário
if prompt := st.chat_input("O que você gostaria de perguntar?"):
    # Adicione a mensagem do usuário ao histórico de chat
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Exiba a mensagem do usuário no contêiner de mensagem de chat
    with st.chat_message("user"):
        st.markdown(prompt)

    # Adicione uma mensagem de "Carregando..."
    with st.spinner('Processando sua pergunta...'):
        try:
            # Chame a função do backend com a entrada do usuário
            response = get_index_name(prompt)
        except Exception as e:
            st.error(f"Ocorreu um erro: {e}")
            st.stop()

    # Exiba a resposta do assistente no contêiner de mensagem de chat
    with st.chat_message("assistant"):
        st.write(response)

    # Adicione a resposta do assistente ao histórico de chat
    st.session_state.messages.append({"role": "assistant", "content": response})

# Botão para abrir a área de upload
if st.toggle("Transcrição de Clientes"):
    # Adicione um botão para enviar um arquivo DOCX
    uploaded_file = st.file_uploader("Envie um arquivo DOCX", type="docx")
    client_name = st.text_input("Nome do Cliente")
    meeting_date = st.date_input("Data da Reunião")

    if uploaded_file and client_name and meeting_date:
        meeting_date_str = meeting_date.strftime("%d/%m/%Y")  # Converter data para string no formato desejado
        with st.spinner('Processando o arquivo...'):
            try:
                index_name = process_docx(uploaded_file, client_name, meeting_date_str)
                st.success(f"Documento processado e indexado com sucesso: {index_name}")
            except Exception as e:
                st.error(f"Ocorreu um erro: {e}")
