import os
import json
from groq import Groq
from dotenv import load_dotenv
from tools import TOOLS

load_dotenv()


# ============================================================
# The system prompt — this IS the agent's brain
# ============================================================

SYSTEM_PROMPT = """You are a research assistant with access to tools.

AVAILABLE TOOLS:
{tool_descriptions}

YOUR RESPONSE MUST ALWAYS BE VALID JSON. NOTHING ELSE.
No explanations. No "Please wait". No fake results. Just JSON.

To call a tool:
{{"thought": "why I am calling this tool", "action": "tool_name", "input": {{"param": "value"}}}}

To give final answer (ONLY after receiving real tool results):
{{"thought": "I have real results from the tools", "action": "finish", "answer": "answer in plain English"}}

STRICT RULES:
1. Output ONLY a single JSON object. Nothing before or after it.
2. Never invent or assume tool results. Wait for real results.
3. Never use finish on the first step.
4. Use vector_search first for document questions.
5. Use compare_documents for any comparison between two countries.
6. Use list_documents when asked what documents exist.
7. Use web_search only when documents have no answer.
8. Maximum 5 steps.
"""


def format_tool_descriptions(tools: dict) -> str:
    """Format tool descriptions for the system prompt."""
    lines = []
    for name, tool in tools.items():
        lines.append(f"- {name}: {tool['description']}")
    return "\n".join(lines)


def parse_agent_response(response_text: str) -> dict:
    """
    Parse the agent's JSON response.
    Handles markdown code blocks and malformed JSON.
    """
    text = response_text.strip()

    # Remove markdown code blocks
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:]
            try:
                return json.loads(part.strip())
            except Exception:
                continue

    # Try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try to extract JSON object
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except Exception:
            pass

    # Fallback — treat as final answer
    return {
        "thought": "Providing direct response",
        "action": "finish",
        "answer": response_text
    }


def execute_tool(action: str, tool_input: dict) -> str:
    """
    Execute a tool and return result as string.
    The agent receives tool results as text.
    """
    if action not in TOOLS:
        return f"Error: Tool '{action}' not found."

    tool_func = TOOLS[action]["function"]

    try:
        result = tool_func(**tool_input)
        return json.dumps(result, indent=2)
    except TypeError as e:
        return json.dumps({
            "success": False,
            "error": f"Wrong parameters: {str(e)}"
        })
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        })


class RAGAgent:
    """
    A ReAct agent that uses tools to answer questions.

    ReAct = Reasoning + Acting
    The agent alternates between:
    - Thinking about what to do
    - Acting (calling a tool)
    - Observing the result
    - Deciding if it needs more info
    """

    def __init__(self, max_steps: int = 5):
        self.client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )
        self.max_steps = max_steps
        self.system_prompt = SYSTEM_PROMPT.format(
            tool_descriptions=format_tool_descriptions(TOOLS)
        )

    def run(self, question: str,
            verbose: bool = True) -> dict:
        """
        Run the agent on a question.
        Returns the final answer and full trace.
        """

        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print(f"{'='*60}")

        messages = [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": question
            }
        ]

        trace = []
        final_answer = None

        for step in range(self.max_steps):
            print(f"\n--- Step {step + 1} ---")

            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.1
            )

            agent_text = response.choices[0].message.content
            parsed = parse_agent_response(agent_text)

            thought = parsed.get("thought", "")
            action = parsed.get("action", "")

            print(f"Thought: {thought}")
            print(f"Action:  {action}")

            # ---- Check if agent is done ----
            if action == "finish":
                final_answer = parsed.get("answer", "")
                trace.append({
                    "step": step + 1,
                    "thought": thought,
                    "action": "finish",
                    "result": final_answer
                })
                print(f"\nFinal Answer:\n{final_answer}")
                break

            # ---- Execute the chosen tool ----
            tool_input = parsed.get("input", {})
            print(f"Input:   {tool_input}")

            observation = execute_tool(action, tool_input)

            # Parse observation for display
            try:
                obs_parsed = json.loads(observation)
                if obs_parsed.get("success"):
                    if "results" in obs_parsed:
                        n = len(obs_parsed["results"])
                        print(f"Result:  ✅ Got {n} results")
                    elif "summary" in obs_parsed:
                        print(f"Result:  ✅ Got summary")
                    elif "comparison" in obs_parsed:
                        print(f"Result:  ✅ Got comparison")
                    elif "documents" in obs_parsed:
                        n = len(obs_parsed["documents"])
                        print(f"Result:  ✅ Found {n} documents")
                else:
                    print(
                        f"Result:  ❌ "
                        f"{obs_parsed.get('error', 'Failed')}"
                    )
            except Exception:
                print(f"Result:  {observation[:100]}")

            # Record step
            trace.append({
                "step": step + 1,
                "thought": thought,
                "action": action,
                "input": tool_input,
                "result": observation
            })

            # Add result to conversation
            messages.append({
                "role": "assistant",
                "content": agent_text
            })
            messages.append({
                "role": "user",
                "content": (
                    f"Tool result:\n{observation}\n\n"
                    "Continue reasoning. If you have enough "
                    "information, provide your final answer "
                    "using the finish action. "
                    "Otherwise, use another tool."
                )
            })

        if final_answer is None:
            final_answer = (
                "I reached the maximum number of steps "
                "without finding a complete answer. "
                "Please try rephrasing your question."
            )

        return {
            "question": question,
            "answer": final_answer,
            "steps": len(trace),
            "trace": trace
        }


# ---- Test the agent ----
if __name__ == "__main__":

    agent = RAGAgent(max_steps=5)

    # Test 1: Simple document question
    result = agent.run(
        "What are the main religions practiced in Japan?"
    )
    print(f"\n✅ Completed in {result['steps']} steps")
    print("\n" + "="*60)

    # Test 2: Comparison question
    result = agent.run(
        "How does religion differ between Japan and India?"
    )
    print(f"\n✅ Completed in {result['steps']} steps")
    print("\n" + "="*60)

    # Test 3: Overview question
    result = agent.run(
        "What documents do you have and what is the "
        "japan document about?"
    )
    print(f"\n✅ Completed in {result['steps']} steps")
    print("\n" + "="*60)

    # Test 4: Out of scope
    result = agent.run(
        "What are the latest Japanese pop culture "
        "trends in 2024?"
    )
    print(f"\n✅ Completed in {result['steps']} steps")