# ASSIGNMENT 25 - PROMPTING & CHAINS

from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.llms import Ollama
from pydantic import BaseModel
from langchain.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

llm = Ollama(model="llama3")

print("===== PART 1: PROMPT TEMPLATE =====")

prompt = PromptTemplate(
    input_variables=["question"],
    template="Answer this question clearly: {question}"
)

print(llm.invoke(prompt.format(question="What is AI?")))

print("===== TASK 2: CHAT PROMPT =====")

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    ("human", "{question}")
])

messages = chat_prompt.format_messages(question="Explain Machine Learning")
print(llm.invoke(messages))

print("===== PART 2: PYDANTIC =====")

class Answer(BaseModel):
    answer: str
    confidence: float
    source: str

parser = PydanticOutputParser(pydantic_object=Answer)

format_instructions = parser.get_format_instructions()

structured_prompt = PromptTemplate(
    input_variables=["question"],
    template="Answer: {question}\n{format_instructions}",
    partial_variables={"format_instructions": format_instructions}
)

response = llm.invoke(structured_prompt.format(question="What is AI?"))
print(response)

print("===== PART 3: SIMPLE CHAIN =====")

simple_chain = prompt | llm
print(simple_chain.invoke({"question": "What is Deep Learning?"}))

print("===== CONDITIONAL CHAIN =====")

def conditional_logic(question):
    if "what" in question.lower():
        return llm.invoke(question)
    else:
        return "Not factual"

print(conditional_logic("What is Python?"))
print(conditional_logic("Tell me a joke"))

print("===== PARALLEL CHAIN =====")

parallel_chain = RunnableParallel(
    answer=lambda x: llm.invoke(x),
    summary=lambda x: llm.invoke("Summarize: " + x),
    followup=lambda x: llm.invoke("Follow-ups: " + x)
)

print(parallel_chain.invoke("What is AI?"))

print("===== RUNNABLE =====")

runnable = RunnablePassthrough() | llm
print(runnable.invoke("Explain Neural Networks"))

print("===== LCEL =====")

lcel = prompt | llm
print(lcel.invoke({"question": "Explain Data Science"}))

