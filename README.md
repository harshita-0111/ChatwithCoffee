# ChatWithCoffee

![ChatWithCoffee Cover](assets/coverimg.png)

ChatWithCoffee is a lightweight, Streamlit-based chatbot assistant powered by Groq + LangChain + LangGraph.
It is designed for a clean local development workflow: simple UI, stateful chat history, and quick iteration.

## Highlights

- Streamlit chat interface with custom coffee avatar
- Groq model integration via `langchain-groq`
- LangGraph-based response flow in the backend
- Session-based chat history with `New Chat` reset
- Environment-variable-based secret management (`.env`)

## Project Structure

```text
Chatwithcoffee/
├─ frontend.py          # Streamlit UI
├─ backend.py           # LLM + LangGraph chat logic
├─ requirements.txt     # Python dependencies
├─ .env                 # Local secrets (ignored by git)
└─ assets/
	 ├─ coverimg.png
	 ├─ cover.svg
	 └─ images.jpeg       # Assistant avatar
```

## Prerequisites

- Python 3.10+ (project currently tested on Python 3.13)
- Git
- A Groq API key

## Setup

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

3. Create a `.env` file in the project root:

```bash
GROQ_API_KEY=your_groq_api_key_here
```

## Run

Start the Streamlit app:

```bash
streamlit run frontend.py
```

Then open the local URL shown in your terminal (usually `http://localhost:8501`).

## How It Works

1. `frontend.py` captures user input and keeps chat history in `st.session_state`.
2. User prompt and prior history are passed to `generate_reply(...)` in `backend.py`.
3. `backend.py` builds message objects, runs a minimal LangGraph flow, and calls Groq.
4. Assistant response is returned to the UI and rendered with the coffee avatar.

## Security Notes

- Do not commit `.env`.
- Rotate API keys immediately if they are ever exposed in commit history.
- Keep secrets in environment variables only.

## Troubleshooting

- `Missing GROQ_API_KEY in environment (.env)`
	- Ensure `.env` exists and contains `GROQ_API_KEY=...`.
- `ModuleNotFoundError`
	- Activate `.venv` and run `pip install -r requirements.txt` again.
- Streamlit opens but no model response
	- Verify internet connectivity and Groq key validity.

## Tech Stack

- Streamlit
- LangChain
- LangGraph
- Groq
- Python Dotenv


