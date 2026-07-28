import os
from typing import Annotated, TypedDict, Iterator

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from duckduckgo_search import DDGS

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
        try:
            import streamlit as st
            if "GROQ_API_KEY" in st.secrets:
                api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            pass

    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY in environment (.env or Streamlit Secrets)")

    return ChatGroq(model="llama-3.1-8b-instant", api_key=api_key)


# --- Workspace Tools Setup ---
# Restrict operations to the project workspace root path.
WORKSPACE_ROOT = os.path.abspath(os.path.dirname(__file__))


def _resolve_path(path: str) -> str:
    # Resolve the absolute path and ensure it is inside the workspace root.
    abs_path = os.path.abspath(os.path.join(WORKSPACE_ROOT, path))
    if not abs_path.startswith(WORKSPACE_ROOT):
        raise ValueError("Security violation: path is outside the workspace directory.")
    return abs_path


@tool
def list_workspace_files() -> str:
    """Lists all files in the current workspace directory, excluding hidden folders/files."""
    try:
        files_list = []
        for root, dirs, files in os.walk(WORKSPACE_ROOT):
            # Exclude hidden directories like .git, .venv, etc.
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for f in files:
                if f.startswith('.'):
                    continue
                # Get path relative to workspace root
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, WORKSPACE_ROOT)
                files_list.append(rel_path)
        if not files_list:
            return "Workspace is empty."
        return "\n".join(files_list)
    except Exception as e:
        return f"Error listing files: {str(e)}"


@tool
def read_workspace_file(file_path: str) -> str:
    """Reads the contents of a specific file inside the workspace.
    
    Args:
        file_path: The relative path to the file from the workspace root.
    """
    try:
        target = _resolve_path(file_path)
        if not os.path.exists(target):
            return f"Error: File '{file_path}' does not exist."
        if os.path.isdir(target):
            return f"Error: '{file_path}' is a directory. Use list_workspace_files to see its content."
        with open(target, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file '{file_path}': {str(e)}"


@tool
def write_workspace_file(file_path: str, content: str) -> str:
    """Writes (or overwrites) a file in the workspace with the specified content.
    
    Args:
        file_path: The relative path to the file from the workspace root.
        content: The text content to write to the file.
    """
    try:
        target = _resolve_path(file_path)
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote to file '{file_path}'."
    except Exception as e:
        return f"Error writing file '{file_path}': {str(e)}"


@tool
def web_search(query: str) -> str:
    """Searches the web for the given query to retrieve up-to-date information.
    
    Args:
        query: The search query.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=4))
        if not results:
            return "No search results found."
        formatted = []
        for r in results:
            formatted.append(f"Title: {r.get('title')}\nURL: {r.get('href')}\nSnippet: {r.get('body')}\n")
        return "\n---\n".join(formatted)
    except Exception as e:
        return f"Web search error: {str(e)}"


tools = [list_workspace_files, read_workspace_file, write_workspace_file, web_search]


def _chat_node(state: ChatState) -> ChatState:
    # This node calls the model with tools enabled and returns the new assistant message.
    llm = _build_llm()
    llm_with_tools = llm.bind_tools(tools)
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def _route(state: ChatState):
    # Route to tools if the last message has tool calls, else end.
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END


# --- Graph Setup ---
def _build_graph():
    # ReAct-style graph: START -> chat_node -> tools -> chat_node ...
    graph = StateGraph(ChatState)
    graph.add_node("chat_node", _chat_node)
    graph.add_node("tools", ToolNode(tools))
    
    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", _route, ["tools", END])
    graph.add_edge("tools", "chat_node")
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
    # Stream the chatbot's response, yielding tool logs as they execute,
    # and streaming final text tokens as they arrive.
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
    llm_with_tools = llm.bind_tools(tools)
    
    current_messages = list(messages)
    
    while True:
        response = llm_with_tools.invoke(current_messages)
        current_messages.append(response)
        
        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                yield f"⚙️ *Assisting Agent: Calling Tool* `{tool_name}` *with args:* `{tool_args}`...\n"
                
                # Run tool
                tool_fn = next((t for t in tools if t.name == tool_name), None)
                if tool_fn:
                    try:
                        tool_result = tool_fn.invoke(tool_args)
                        yield f"💡 *Tool `{tool_name}` returned result:*\n{tool_result[:300]}...\n\n"
                        current_messages.append(ToolMessage(content=tool_result, tool_call_id=tool_call["id"], name=tool_name))
                    except Exception as e:
                        error_msg = f"Error executing tool '{tool_name}': {str(e)}"
                        yield f"❌ *{error_msg}*\n\n"
                        current_messages.append(ToolMessage(content=error_msg, tool_call_id=tool_call["id"], name=tool_name))
                else:
                    error_msg = f"Error: Tool '{tool_name}' not found."
                    yield f"❌ *{error_msg}*\n\n"
                    current_messages.append(ToolMessage(content=error_msg, tool_call_id=tool_call["id"], name=tool_name))
        else:
            # No more tool calls, stream the final response text
            current_messages.pop()  # remove the final empty response message to stream it
            for chunk in llm.stream(current_messages):
                text = getattr(chunk, "text", getattr(chunk, "content", ""))
                if text:
                    yield str(text)
            break