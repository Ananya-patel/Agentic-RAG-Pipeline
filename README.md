# ◆ Agentic RAG System

> Project 5 of my RAG Mastery Journey

A fully agentic RAG system that reasons step by step before
answering. The agent decides which tools to use, executes them,
reflects on the results, and generates grounded answers —
just like a research assistant that thinks before it speaks.

---

##  What Makes This "Agentic"

Every previous RAG project followed a fixed pipeline:
```
Question → retrieve → generate → answer
```

This project uses a ReAct reasoning loop:
```
Question
  ↓
Thought: "What do I need to answer this?"
  ↓
Action: pick the right tool
  ↓
Observation: read tool result
  ↓
Thought: "Do I have enough? What next?"
  ↓
Repeat until confident → Final Answer
```

The LLM is no longer just a text generator.
It's a reasoning engine that plans and executes.

---

##  Tools The Agent Has

| Tool | When Used |
|---|---|
| `vector_search` | Find relevant chunks from documents |
| `compare_documents` | Compare two docs on a topic |
| `summarize_document` | Get overview of one document |
| `list_documents` | See what's available |
| `web_search` | Fallback for out-of-scope questions |

---

##  Architecture
```
tools.py    → 5 tools the agent can call
agent.py    → ReAct reasoning loop (Thought→Action→Observation)
memory.py   → Short-term + long-term conversation memory
app.py      → Streamlit chat UI with agent trace
```

---

##  What's Indexed

| Document | Chunks |
|---|---|
| japan_culture.pdf | 102 |
| india_culture.pdf | 241 |
| france_culture.pdf | 167 |
| **Total** | **510** |

---

##  Key Concepts Learned

| Concept | What it means |
|---|---|
| ReAct pattern | Reasoning + Acting in a loop |
| Tool use | LLM decides which function to call |
| Agent trace | Full visibility into reasoning steps |
| Conversation memory | Short-term + long-term context |
| Model selection | 70B needed for reliable tool use |
| Prompt as brain | System prompt shapes all decisions |

---

##  Setup & Run Locally

**1. Clone and install**
```bash
git clone https://github.com/YOUR_USERNAME/agentic-rag.git
cd agentic-rag
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**2. Environment variables**
```
GROQ_API_KEY=your-groq-key-here
```

**3. Add PDFs and build ChromaDB**
```bash
# Copy your PDFs into the folder then:
python ingest.py
```

**4. Run**
```bash
streamlit run app.py
```

---

##  Project Structure
```
project5/
├── tools.py      # 5 agent tools
├── agent.py      # ReAct reasoning loop
├── memory.py     # Conversation memory
├── ingest.py     # ChromaDB ingestion
├── app.py        # Streamlit chat UI
├── requirements.txt
└── README.md
```

---

##  RAG Mastery Journey

| Project | Topic | Status |
|---|---|---|
| Project 1 | Document Analysis Using LLMs | ✅ Complete |
| Project 2 | RAG System From Scratch | ✅ Complete |
| Project 3 | Multi-Document RAG | ✅ Complete |
| Project 4 | GraphRAG Pipeline | ✅ Complete |
| **Project 5** | **Agentic RAG System** | ✅ **Complete** |
| Project 6 | Real-Time RAG Assistant | 🔄 Next |

---

