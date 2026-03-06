import streamlit as st
from agent import RAGAgent
from memory import ConversationMemory
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="RAG Agent",
    page_icon="◆",
    layout="wide"
)

# ---- Session state ----

if "agent" not in st.session_state:
    st.session_state.agent = RAGAgent(max_steps=5)

if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory(
        max_short_term=10
    )

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---- Sidebar ----

with st.sidebar:
    st.title("◆ RAG Agent")
    st.caption("Agentic retrieval across 3 documents")
    st.divider()

    st.subheader("📄 Documents")
    st.markdown("- japan_culture.pdf — 102 chunks")
    st.markdown("- india_culture.pdf — 241 chunks")
    st.markdown("- france_culture.pdf — 167 chunks")
    st.divider()

    st.subheader("🧠 Memory")
    mem = st.session_state.memory.summary()
    st.metric("Total turns", mem["total_turns"])
    st.metric("Facts stored", mem["long_term_facts"])
    st.metric("In context",   mem["short_term_turns"])
    st.divider()

    st.subheader("⚙️ Settings")
    max_steps = st.slider("Max reasoning steps", 2, 7, 5)
    st.session_state.agent.max_steps = max_steps
    show_trace = st.toggle("Show agent reasoning", value=True)

    if st.button("💾 Save session"):
        st.session_state.memory.save_session()
        st.success("Session saved!")

    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.session_state.memory = ConversationMemory(
            max_short_term=10
        )
        st.rerun()


# ---- Main area ----

st.title("RAG Agent")
st.caption(
    "Ask questions across Japan, India and France culture. "
    "The agent reasons step by step and shows its work."
)
st.divider()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

        if msg["role"] == "assistant" and show_trace:
            trace = msg.get("trace", [])
            tool_steps = [
                s for s in trace
                if s.get("action") != "finish"
            ]
            if tool_steps:
                with st.expander(
                    f"🔍 Agent reasoning "
                    f"({len(tool_steps)} steps)"
                ):
                    for step in tool_steps:
                        action = step.get("action", "")
                        thought = step.get("thought", "")
                        inp = step.get("input", {})

                        st.markdown(f"**Tool:** `{action}`")
                        if inp:
                            st.markdown(
                                f"**Input:** `{inp}`"
                            )
                        st.caption(f"💭 {thought}")
                        st.divider()


# ---- Chat input ----

query = st.chat_input(
    "Ask anything about your documents..."
)

if query:
    # Show user message
    with st.chat_message("user"):
        st.write(query)

    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    # Add memory context
    memory_context = (
        st.session_state.memory.get_full_context()
    )
    full_question = query
    if memory_context:
        full_question = (
            f"{memory_context}\n\nCurrent question: {query}"
        )

    # Run agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = st.session_state.agent.run(
                full_question, verbose=False
            )

        answer = result["answer"]
        trace  = result["trace"]
        steps  = result["steps"]

        st.write(answer)
        st.caption(f"Completed in {steps} reasoning steps")

        tool_steps = [
            s for s in trace
            if s.get("action") != "finish"
        ]
        if show_trace and tool_steps:
            with st.expander(
                f"🔍 Agent reasoning ({len(tool_steps)} steps)"
            ):
                for step in tool_steps:
                    action = step.get("action", "")
                    thought = step.get("thought", "")
                    inp = step.get("input", {})

                    st.markdown(f"**Tool:** `{action}`")
                    if inp:
                        st.markdown(f"**Input:** `{inp}`")
                    st.caption(f"💭 {thought}")
                    st.divider()

    # Update memory
    st.session_state.memory.add_turn(
        question=query,
        answer=answer,
        steps=steps
    )

    if any(c.isdigit() for c in answer):
        st.session_state.memory.add_fact(
            fact=answer[:150].replace("\n", " "),
            source="agent_answer"
        )

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "trace": trace,
        "steps": steps
    })

    st.rerun()
