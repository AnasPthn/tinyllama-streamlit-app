from langchain_openai import ChatOpenAI #type: ignore
from dotevn import load_dotevn

load_dotevn()

model = ChatOpenAI(model='gpt-4')

result = model.invoke("Who is precident of USA")