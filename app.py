from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from config.config import jolly_llm


def assistant(user_msg:str):

    sys_prompt = """
Your name is Jolly, a friendly and helpful assistant. You are here to assist the user with their queries and provide information in a friendly manner. Always respond in a positive and cheerful tone."""

    prompt = ChatPromptTemplate.from_messages([("system", sys_prompt), ("human", "{input}")])

    chain = prompt | jolly_llm()

    response = chain.invoke({"input": user_msg})

    return response.content


if __name__ == "__main__":
    user_input = input("User: ")
    response = assistant(user_input)
    print(f"Jolly: {response}")