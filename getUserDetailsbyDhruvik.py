from langchain.schema import SystemMessage, HumanMessage
from config.config import openai_llm
from langchain_core.prompts import ChatPromptTemplate
# from langgraph.prebuit import ToolNode, tools_condition

from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver

from typing import List, Annotated, Dict, Any, Optional, Union
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage

jolly_llm = openai_llm()

class JollyState(TypedDict, total=False):
    """
    list of messages of both user and assistant
    """
    messages: Annotated[list[AnyMessage], add_messages]

assistant_prompt = """
You are Jolly, a smart and friendly assistant with a curious mind. 
Your goal is to help users with their questions while gently and creatively asking for their name, email, and contact number information—without making them feel like you’re a detective. 
Always keep your tone warm, positive, and inquisitive. 
Start every conversation by warmly introducing yourself and asking for these details in a casual, natural way, so users feel comfortable sharing. 
Then, assist them with their queries in a helpful and engaging manner.
"""

def assistant_node(state: JollyState):
    all_messages = state.get('messages', [])

    username = state.get("Username", "User")
    email = state.get("EmailId", "N/A")
    number = state.get("PhoneNumber", "N/A")

    personalized_prompt = assistant_prompt.format(
        username=username,
        email=email,
        number=number
    )

    messages = [SystemMessage(content=personalized_prompt)] + all_messages
    llm_response = jolly_llm.invoke(messages)
    # print(f"LLM Response: {llm_response}")

    return {
        "messages": llm_response
    }

jolly_graph = StateGraph(JollyState)

jolly_graph.add_node('assistant', assistant_node)
jolly_graph.add_edge(START, 'assistant')
jolly_graph.add_edge('assistant', END)

jolly_ai = jolly_graph.compile(checkpointer=MemorySaver())


def process_message(message: str):
    user_msg = [HumanMessage(content=message)]
    config = {"configurable": {"thread_id": "1234"}}
    try:
        response = jolly_ai.invoke({"messages": user_msg}, config=config)
    except Exception as e:
        print(f"error caught: {str(e)}")
        raise e
        
    return response['messages'][-1].content


if __name__ == "__main__":
    print("Welcome to jolly Assistant !")
    print("type 'exit' to end the conversation.")
    
    while True:
        user_msg = input("User :")
        if user_msg.lower() == 'exit':
            print("bye !")
            break
        
        response = process_message(user_msg)
        print(f"\nAssistant : {response}\n")
