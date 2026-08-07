"""
app.py
Streamlit UI for the AI Coding Assistant.
Wires together: classifier -> router -> RAG -> relevance check ->
code generation/explanation -> code execution -> feedback learning -> memory.
"""

import streamlit as st
from modules.memory import init_memory, add_turn, get_context_string, update_preferences, set_last_code
from modules.code_runner import run_code, extract_code_block
from modules.rag import add_document
import time
from modules.router import route_request, handle_explain

st.set_page_config(page_title="AI Coding Assistant", page_icon="🤖", layout="wide")

# ---------------------------------------------------------
# Session state init
# ---------------------------------------------------------
if "memory" not in st.session_state:
    st.session_state.memory = init_memory()

if "messages" not in st.session_state:
    st.session_state.messages = []  # for rendering chat bubbles

if "pending_feedback" not in st.session_state:
    st.session_state.pending_feedback = False

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

if "last_code_block" not in st.session_state:
    st.session_state.last_code_block = None

if "last_uploaded_name" not in st.session_state:
    st.session_state.last_uploaded_name = None

# ---------------------------------------------------------
# Sidebar: memory visualization + preferences + file upload
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    lang = st.selectbox("Preferred language", ["Unspecified", "Python", "JavaScript", "Java", "C++"])
    framework = st.selectbox("Preferred framework", ["Unspecified", "None", "Flask", "Django", "React", "PyTorch", "TensorFlow"])
    update_preferences(
        st.session_state.memory,
        language=None if lang == "Unspecified" else lang,
        framework=None if framework == "Unspecified" else framework,
    )

    st.divider()
    uploaded_file = st.file_uploader("Upload a code file (for explanation)", type=["py", "js", "java", "cpp", "txt"])

    if uploaded_file:
        uploaded_code = uploaded_file.read().decode("utf-8")
        st.success(f"✅ Loaded {uploaded_file.name} ({len(uploaded_code)} chars)")
        with st.expander("Preview uploaded file"):
            st.code(uploaded_code, language="python")
    else:
        uploaded_code = ""

    st.divider()
    st.subheader("🧠 Memory")
    st.caption(f"Turns stored: {len(st.session_state.memory['history'])}")
    with st.expander("View conversation memory"):
        st.json(st.session_state.memory)


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.title("🤖 AI Coding Assistant")
st.caption("Explain code, generate code with RAG-backed knowledge, execute it, and teach it new solutions.")

# ---------------------------------------------------------
# Render chat history
# ---------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------------------------------
# Auto-explain newly uploaded files
# ---------------------------------------------------------
if uploaded_file and uploaded_file.name != st.session_state.last_uploaded_name:
    st.session_state.last_uploaded_name = uploaded_file.name

    auto_query = f"Explain this uploaded file: {uploaded_file.name}"
    st.session_state.messages.append({"role": "user", "content": auto_query})
    with st.chat_message("user"):
        st.markdown(auto_query)

    with st.chat_message("assistant"):
        with st.spinner("Reading and explaining your file..."):
            result = handle_explain(auto_query, code_snippet=uploaded_code)
        st.markdown(result["answer"])

    add_turn(st.session_state.memory, "user", auto_query, intent="explain")
    add_turn(st.session_state.memory, "assistant", result["answer"], intent="explain")
    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})

# ---------------------------------------------------------
# Chat input
# ---------------------------------------------------------
user_input = st.chat_input("Ask me to explain or generate code...")

if user_input:
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()

        # -------------------------------------------------
        # CASE 1: We're waiting for the user to supply a solution
        # after a failed relevance check (Step 7 of spec)
        # -------------------------------------------------
        if st.session_state.pending_feedback:
            doc_id = f"user_feedback_{int(time.time())}"
            add_document(
                text=user_input,
                metadata={
                    "source": "user_feedback",
                    "original_query": st.session_state.pending_query,
                },
                doc_id=doc_id,
            )
            response_text = (
                "✅ Thanks! I've stored that solution in my knowledge base "
                "(Chroma) so I can use it for similar requests in the future."
            )
            placeholder.markdown(response_text)

            add_turn(st.session_state.memory, "user", user_input)
            add_turn(st.session_state.memory, "assistant", response_text)

            st.session_state.pending_feedback = False
            st.session_state.pending_query = None
            st.session_state.messages.append({"role": "assistant", "content": response_text})

        # -------------------------------------------------
        # CASE 2: Normal request -> classify -> route
        # -------------------------------------------------
        else:
            # Give the LLM conversation context (spec item 8)
            context_str = get_context_string(st.session_state.memory)
            full_query = f"{context_str}\n\nCurrent request: {user_input}" if context_str else user_input

            code_snippet = uploaded_code  # empty string if nothing uploaded
            result = route_request(full_query, code_snippet=code_snippet)

            # Simulated streaming: reveal the answer progressively
            answer = result["answer"]
            displayed = ""
            for chunk in answer.split(" "):
                displayed += chunk + " "
                placeholder.markdown(displayed + "▌")
                time.sleep(0.01)
            placeholder.markdown(displayed)

            add_turn(st.session_state.memory, "user", user_input, intent=result.get("intent"))
            add_turn(st.session_state.memory, "assistant", answer, intent=result.get("intent"))
            st.session_state.messages.append({"role": "assistant", "content": answer})

            if result["type"] == "needs_feedback":
                st.session_state.pending_feedback = True
                st.session_state.pending_query = user_input
                st.info("💡 Reply with the correct solution in the chat box to teach me.")

            elif result["type"] == "generated_code":
                code = extract_code_block(answer)
                st.session_state.last_code_block = code
                set_last_code(st.session_state.memory, code)
                if result.get("sources"):
                    st.caption(f"📚 Sources used: {', '.join(result['sources'])}")


# ---------------------------------------------------------
# Execute button — always shows the most recently generated code
# ---------------------------------------------------------
if st.session_state.last_code_block:
    st.divider()
    st.subheader("▶️ Run last generated code")
    st.code(st.session_state.last_code_block, language="python")

    if st.button("Execute Code"):
        with st.spinner("Running..."):
            result = run_code(st.session_state.last_code_block)

        if result["success"]:
            st.success("Execution succeeded")
        else:
            st.error("Execution failed")

        if result["stdout"]:
            st.text("STDOUT:")
            st.code(result["stdout"])
        if result["stderr"]:
            st.text("STDERR:")
            st.code(result["stderr"])