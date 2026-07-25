import os
from typing import Annotated, TypedDict, Iterator

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

# --- Configuration ---
# Load variables from .env so GROQ_API_KEY is available at runtime.
load_dotenv()

SYSTEM_PROMPT = "You are ChatWithCoffee, a helpful and concise chatbot assistant."


class ChatState(TypedDict):
    # Conversation messages are the shared state passed through the graph.
    messages: Annotated[list[BaseMessage], add_messages]


# --- Model Setup ---
def _build_llm() -> ChatGroq:
    # Keep key validation here so failures are clear and early.
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY in environment (.env)")

    return ChatGroq(model="llama-3.1-8b-instant", api_key=api_key)


def _chat_node(state: ChatState) -> ChatState:
    # This node calls the model once and returns the new assistant message.
    llm = _build_llm()
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# --- Graph Setup ---
def _build_graph():
    # Minimal one-step graph: START -> chat_node -> END.
    graph = StateGraph(ChatState)
    graph.add_node("chat_node", _chat_node)
    graph.add_edge(START, "chat_node")
    graph.add_edge("chat_node", END)
    return graph.compile()


chatbot = _build_graph()


# --- Public Chat API ---
def generate_reply(history: list[dict[str, str]], user_input: str) -> str:
    # Start each request with a system instruction for response style.
    messages: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]

    for item in history:
        role = item.get("role")
        content = item.get("content", "")
        if not content:
            continue
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    # Add the new user question, then execute the graph.
    messages.append(HumanMessage(content=user_input))
    result = chatbot.invoke({"messages": messages})
    last_message = result["messages"][-1]
    return str(last_message.content)


def generate_reply_stream(history: list[dict[str, str]], user_input: str) -> Iterator[str]:
    # Build the messages list same as generate_reply, but stream the model output.
    messages: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]

    for item in history:
        role = item.get("role")
        content = item.get("content", "")
        if not content:
            continue
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=user_input))

    llm = _build_llm()
    # `ChatGroq.stream` yields incremental chunks; yield their text content.
    for chunk in llm.stream(messages):
        text = getattr(chunk, "text", None)
        if text is None:
            # Fallback to `content` if available.
            text = getattr(chunk, "content", None)
        if not text:
            continue
        yield str(text)