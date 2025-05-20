from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from config.config import openai_llm

from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from typing import List, Annotated, Dict, Any, Optional, Union
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage

jolly_llm = openai_llm()

class JollyState(TypedDict, total=False):
    """
    State for the Jolly assistant.

    messages: List of messages exchanged between the user and the assistant.
    """
       
    messages: Annotated[list[AnyMessage], add_messages]


assistant_prompt = """
your name is jolly a friendly assitant.you have to collect the user name, email, phone number remember always ask users in a creative and fun way don't make conversation like a robot.
"""


def assistant_node(state: JollyState):

    all_messages = state.get('messages', [])

    messages = [SystemMessage(content=assistant_prompt)] + all_messages

    llm_response = jolly_llm.invoke(messages)

        
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
        response = jolly_ai.invoke(
            {"messages": user_msg}, config=config
        )

    except Exception as e:
        print(f"❌ Error in Graph Invoke: {str(e)}")
        raise e
    
    return response['messages'][-1].content


if __name__ == "__main__":
    print("Welcome to Jolly Assistant!")
    print("Type 'exit' to end the conversation.")
    while True:
        user_msg = input("User: ")
        if user_msg.lower() == 'exit':
            print("Goodbye!")
            break

        response = process_message(user_msg)
        print(f"\nAssistant: {response}\n")
