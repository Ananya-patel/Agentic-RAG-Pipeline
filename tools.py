import os
import chromadb
import PyPDF2
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pathlib import Path
from ddgs import DDGS


load_dotenv()

# ---- Load shared components once ----
# These are expensive to load so we do it once
# and reuse across all tool calls

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path="chroma_db")

try:
    collection = chroma_client.get_collection("documents")
    print(f"Loaded collection: {collection.count()} chunks")
except Exception:
    collection = None
    print("No ChromaDB collection found — ingest PDFs first")


# ============================================================
# TOOL 1: Vector Search
# ============================================================

def vector_search(query: str, top_k: int = 4) -> dict:
    """
    Search indexed documents for relevant text chunks.

    Use this tool when:
    - User asks about a specific topic
    - You need factual information from the documents
    - You need to find what documents say about something

    Returns chunks with their source and similarity score.
    """
    if collection is None:
        return {
            "success": False,
            "error": "No documents indexed. Run ingest.py first.",
            "results": []
        }

    try:
        query_embedding = model.encode([query]).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        chunks = []
        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i]
            similarity = round(1 / (1 + distance), 4)

            chunks.append({
                "text": results["documents"][0][i],
                "source": results["metadatas"][0][i]["source"],
                "page": results["metadatas"][0][i]["page"],
                "similarity": similarity
            })

        return {
            "success": True,
            "query": query,
            "results": chunks,
            "total_found": len(chunks)
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "results": []
        }


# ============================================================
# TOOL 2: Document Summary
# ============================================================

def summarize_document(doc_id: str) -> dict:
    """
    Get a summary of a specific document.

    Use this tool when:
    - User asks "what is document X about?"
    - User wants an overview of a specific file
    - You need context about what a document covers

    doc_id options: japan_culture, india_culture, france_culture
    """
    if collection is None:
        return {"success": False, "error": "No documents indexed"}

    try:
        # Get chunks from this specific document
        results = collection.get(
            where={"doc_id": doc_id},
            limit=5,
            include=["documents"]
        )

        if not results["ids"]:
            return {
                "success": False,
                "error": f"Document '{doc_id}' not found.",
                "available": ["japan_culture",
                              "india_culture",
                              "france_culture"]
            }

        # Use first 5 chunks as representative sample
        sample_text = "\n\n".join(
            results["documents"][:5]
        )

        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{
                "role": "user",
                "content": f"""Summarize this document in 
3-4 sentences. Be specific about the main topics covered.

Document sample:
{sample_text}

Summary:"""
            }]
        )

        return {
            "success": True,
            "doc_id": doc_id,
            "summary": response.choices[0].message.content,
            "chunks_available": collection.count()
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================
# TOOL 3: Compare Documents
# ============================================================

def compare_documents(doc1: str, doc2: str, topic: str) -> dict:
    doc1 = doc1.replace(".pdf", "").replace(" ", "_")
    doc2 = doc2.replace(".pdf", "").replace(" ", "_")
    """
    Compare two documents on a specific topic.

    Use this tool when:
    - User asks how X differs between two countries/documents
    - User wants to compare two documents on a topic
    - Questions contain words like "compare", "differ",
      "versus", "between X and Y"

    Example: compare_documents("japan_culture",
                               "india_culture", "religion")
    """
    if collection is None:
        return {"success": False, "error": "No documents indexed"}

    try:
        query_embedding = model.encode([topic]).tolist()

        # Get relevant chunks from each document separately
        def get_doc_chunks(doc_id):
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=3,
                where={"doc_id": doc_id},
                include=["documents"]
            )
            return "\n".join(results["documents"][0])

        doc1_text = get_doc_chunks(doc1)
        doc2_text = get_doc_chunks(doc2)

        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{
                "role": "user",
                "content": f"""Compare {doc1} and {doc2} 
specifically on the topic of: {topic}

Use ONLY the information provided below.

{doc1} content:
{doc1_text}

{doc2} content:
{doc2_text}

Provide a structured comparison highlighting key differences
and similarities."""
            }]
        )

        return {
            "success": True,
            "doc1": doc1,
            "doc2": doc2,
            "topic": topic,
            "comparison": response.choices[0].message.content
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================
# TOOL 4: List Available Documents
# ============================================================

def list_documents() -> dict:
    """
    List all documents currently indexed in the database.

    Use this tool when:
    - User asks "what documents do you have?"
    - You need to know what's available before searching
    - User asks about a topic and you're unsure which
      document has it
    """
    if collection is None:
        return {"success": False, "error": "No documents indexed"}

    try:
        all_items = collection.get(
            include=["metadatas"]
        )

        from collections import Counter
        doc_counts = Counter(
            m["doc_id"] for m in all_items["metadatas"]
        )

        documents = []
        for doc_id, count in doc_counts.most_common():
            documents.append({
                "doc_id": doc_id,
                "chunks": count,
                "filename": f"{doc_id}.pdf"
            })

        return {
            "success": True,
            "total_chunks": collection.count(),
            "documents": documents
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================
# TOOL 5: Web Search
# ============================================================

def web_search(query: str, max_results: int = 3) -> dict:
    """
    Search the web for current information.

    Use this tool when:
    - The question cannot be answered from indexed documents
    - User asks about current events or recent information
    - You need to verify or supplement document information
    - Documents don't contain the answer

    Only use this as a FALLBACK after trying vector_search first.
    """
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(
                query,
                max_results=max_results
            ):
                results.append({
                    "title": r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "url": r.get("href", "")
                })

        if not results:
            return {
                "success": False,
                "error": "No web results found",
                "results": []
            }

        return {
            "success": True,
            "query": query,
            "results": results,
            "total_found": len(results)
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "results": []
        }


# ============================================================
# Tool Registry — what the agent can see
# ============================================================

TOOLS = {
    "vector_search": {
        "function": vector_search,
        "description": (
            "Search indexed documents for relevant information. "
            "Input: query string. "
            "Use for: finding facts, answering topic questions."
        )
    },
    "summarize_document": {
        "function": summarize_document,
        "description": (
            "Get a summary of a specific document. "
            "Input: doc_id (japan_culture, india_culture, "
            "france_culture). "
            "Use for: overview questions about one document."
        )
    },
    "compare_documents": {
        "function": compare_documents,
        "description": (
            "Compare two documents on a specific topic. "
            "Input: doc1, doc2, topic. "
            "Use for: comparison questions between countries."
        )
    },
    "list_documents": {
        "function": list_documents,
        "description": (
            "List all available indexed documents. "
            "Input: none. "
            "Use for: when asked what documents are available."
        )
    },
    "web_search": {
        "function": web_search,
        "description": (
            "Search the web for current information. "
            "Input: query string. "
            "Use ONLY as fallback when documents lack the answer."
        )
    }
}


# ---- Test all tools ----
if __name__ == "__main__":
    import json

    print("\n" + "="*50)
    print("TOOL 1: vector_search")
    print("="*50)
    result = vector_search("What is Shinto religion?")
    print(f"Found {result['total_found']} chunks")
    if result["results"]:
        print(f"Top result ({result['results'][0]['similarity']}):")
        print(result["results"][0]["text"][:200])

    print("\n" + "="*50)
    print("TOOL 2: summarize_document")
    print("="*50)
    result = summarize_document("japan_culture")
    if result["success"]:
        print(result["summary"])

    print("\n" + "="*50)
    print("TOOL 3: compare_documents")
    print("="*50)
    result = compare_documents(
        "japan_culture", "india_culture", "religion"
    )
    if result["success"]:
        print(result["comparison"][:400])

    print("\n" + "="*50)
    print("TOOL 4: list_documents")
    print("="*50)
    result = list_documents()
    if result["success"]:
        for doc in result["documents"]:
            print(f"  {doc['doc_id']}: {doc['chunks']} chunks")

    print("\n" + "="*50)
    print("TOOL 5: web_search")
    print("="*50)
    result = web_search("Japanese culture 2024")
    if result["success"]:
        print(f"Found {result['total_found']} web results")
        print(result["results"][0]["title"])
        print(result["results"][0]["snippet"][:200])