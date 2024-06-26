from langchain_aws import ChatBedrock

llm_model = ChatBedrock(
    credentials_profile_name="default",
    region_name="us-east-1",
    model_id='anthropic.claude-3-5-sonnet-20240620-v1:0',
    model_kwargs={
        "max_tokens": 3000,
        "temperature": 0.1,
        "top_p": 0.9
    }
)

# Defina sua pergunta
pergunta = "Qual é o sentido da vida?"

# Use o método query para fazer a pergunta ao modelo
resposta = llm_model.invoke(pergunta)

print(resposta)
