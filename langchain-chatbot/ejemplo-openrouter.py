from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from os import getenv
from dotenv import load_dotenv
from pydantic import SecretStr
import os

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""

load_dotenv()

chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "Eres un asistente útil que ayuda a los desarrolladores a escribir código. "
        "Utiliza buenas prácticas de programación y asegúrate de que el código sea eficiente y legible "
    ),
    HumanMessagePromptTemplate.from_template("{question}")
])

llm = ChatOpenAI(
    api_key=SecretStr(getenv("OPENROUTER_API_KEY", "")),
    base_url="https://openrouter.ai/api/v1",
    model="openai/gpt-oss-20b:free",
)

question = "Escribe un programa en Python que imprima los números del 1 al 10."

prompt_value = chat_prompt.format_prompt(question=question)
messages = prompt_value.to_messages()
print("Messages:", messages)
response_messages = llm.invoke(messages)
print(response_messages.content)
