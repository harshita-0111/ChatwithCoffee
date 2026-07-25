import streamlit as st
import json
import os
import uuid
from typing import Any

from backend import generate_reply, generate_reply_stream

# --- UI Configuration ---
st.title("ChatWithCoffee")
ASSISTANT_AVATAR = "assets/images.jpeg"

SAVED_PATH = "saved_chats.json"


def _load_saved_chats() -> list[dict[str, Any]]:
    if os.path.exists(SAVED_PATH):
        try:
            with open(SAVED_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _save_saved_chats(chats: list[dict[str, Any]]) -> None:
    with open(SAVED_PATH, "w", encoding="utf-8") as f:
        json.dump(chats, f, ensure_ascii=False, indent=2)


# --- Session Initialization ---
# Keep chat history in session_state so it persists across reruns.
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I’m your coffee chatbot assistant."}
    ]

if "saved_chats" not in st.session_state:
    st.session_state.saved_chats = _load_saved_chats()

# Reset history to start a fresh conversation.
if st.button("New Chat"):
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I’m your coffee chatbot assistant."}
    ]
    st.rerun()

# --- Sidebar: Saved Conversations ---
st.sidebar.title("Conversations")

save_title = st.sidebar.text_input("Save current chat as", value="")
if st.sidebar.button("Save Chat"):
    title = save_title.strip() or f"Chat {len(st.session_state.saved_chats) + 1}"
    entry = {"id": str(uuid.uuid4()), "title": title, "messages": st.session_state.messages}
    st.session_state.saved_chats.append(entry)
    _save_saved_chats(st.session_state.saved_chats)
    st.sidebar.success("Saved")

if st.session_state.saved_chats:
    if "menu_open" not in st.session_state:
        st.session_state.menu_open = None
    
    for entry in st.session_state.saved_chats:
        row_cols = st.sidebar.columns([0.7, 0.15])
        row_cols[0].markdown(entry["title"])

        # Menu button that toggles menu state
        if row_cols[1].button("⋮", key=f"menu_{entry['id']}"):
            st.session_state.menu_open = None if st.session_state.menu_open == entry["id"] else entry["id"]
            st.rerun()

        # Show action buttons when menu is open
        if st.session_state.menu_open == entry["id"]:
            col1, col2 = st.sidebar.columns(2)
            if col1.button("Resume", key=f"resume_{entry['id']}", use_container_width=True):
                st.session_state.messages = entry["messages"].copy()
                st.session_state.menu_open = None
                st.rerun()
            if col2.button("Delete", key=f"delete_{entry['id']}", use_container_width=True):
                st.session_state.saved_chats = [c for c in st.session_state.saved_chats if c["id"] != entry["id"]]
                _save_saved_chats(st.session_state.saved_chats)
                st.session_state.menu_open = None
                st.rerun()
else:
    st.sidebar.info("No saved conversations")

# --- Chat History Rendering ---
for message in st.session_state.messages:
    avatar = ASSISTANT_AVATAR if message["role"] == "assistant" else None
    with st.chat_message(message["role"], avatar=avatar):
        st.write(message["content"])

# --- New User Input Handling ---
# chat_input returns a value only when user submits a new message.
if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    try:
        # Exclude the just-added prompt from history to avoid duplicate input.
        history = st.session_state.messages[:-1]
        full_reply = ""
        # Stream the reply into the chat message as chunks arrive.
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            msg_placeholder = st.empty()
            for chunk in generate_reply_stream(history, prompt):
                full_reply += chunk
                msg_placeholder.write(full_reply)
    except Exception as error:
        # Show readable runtime errors in UI instead of crashing the app.
        full_reply = f"Chat error: {error}"

    st.session_state.messages.append({"role": "assistant", "content": full_reply})