# Frontend.py

import streamlit as st

# Verifique se o módulo backend está disponível e importe-o
try:
    import backend
except ModuleNotFoundError:
    st.error('Módulo backend não encontrado. Verifique se está instalado e disponível.')

# Configurações iniciais da página
st.set_page_config(page_title='Bspzinho', layout='wide')

# CSS personalizado para o tema escuro e estilo do chat
# ... (mantenha o CSS como está)

# Título do chatbot com ícone
st.markdown("<h1 style='text-align: center; color: blue;'>Bspzinho 😎</h1>", unsafe_allow_html=True)

# Inicializa a memória do chatbot e o histórico do chat
if 'memory' not in st.session_state:
    st.session_state.memory = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Função para formatar a resposta do chatbot
def format_response(response):
    # Divida a resposta em linhas
    lines = response.split('- **')
    # Remova espaços vazios e linhas em branco
    lines = [line.strip() for line in lines if line.strip()]
    # Formate cada linha como um item de lista HTML
    formatted_lines = ['<li>' + line.replace('**', '') + '</li>' for line in lines]
    # Junte todas as linhas formatadas em uma lista HTML
    return '<ul>' + ''.join(formatted_lines) + '</ul>'

# Container para o histórico do chat
with st.container():
    st.markdown("## Histórico do Chat")
    # Renderiza o histórico do chat
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            st.markdown(f"<div class='message user-message'>{message['text']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='message assistant-message'>{message['text']}</div>", unsafe_allow_html=True)

# Container para a entrada do usuário
with st.container():
    st.markdown("## Faça sua Pergunta")
    
    # Crie o widget com um label não vazio e oculte-o se necessário
    input_text = st.text_input("Digite aqui sua pergunta...", key="user_input", label_visibility="collapsed")

    # Botão para enviar a mensagem
    send_button = st.button('Enviar')

    # Verifica se Enter foi pressionado ou se o botão foi clicado
    if send_button or st.session_state.user_input:
        with st.spinner('Aguarde enquanto a resposta está sendo gerada...'):
            # Adiciona a mensagem do usuário ao histórico do chat
            st.session_state.chat_history.append({"role": "user", "text": input_text})

            # Processa a mensagem do usuário e obtém a resposta do chatbot
            chat_response, st.session_state.memory = backend.utilizar_llm_com_pinecone_e_RAG(input_text, st.session_state.memory)

            # Formata a resposta do chatbot
            formatted_chat_response = format_response(chat_response)

            # Adiciona a resposta formatada do chatbot ao histórico do chat
            st.session_state.chat_history.append({"role": "assistant", "text": formatted_chat_response})

            # Exibe a resposta formatada do chatbot
            st.markdown(f"<div class='message assistant-message'>{formatted_chat_response}</div>", unsafe_allow_html=True)

            # Limpa a caixa de entrada após o envio
            st.session_state.user_input = ""
