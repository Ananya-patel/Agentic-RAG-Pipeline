import json
from datetime import datetime


class ConversationMemory:
    """
    Stores conversation history for the agent.

    Two types of memory:
    1. Short-term: recent messages in the current session
    2. Long-term: key facts the agent has discovered

    Short-term memory = what was said
    Long-term memory = what was learned
    """

    def __init__(self, max_short_term: int = 10):
        # Recent conversation turns
        self.short_term = []

        # Key facts extracted from answers
        self.long_term = []

        # Maximum turns to keep in short-term
        self.max_short_term = max_short_term

        # Full session log
        self.session_log = []

    def add_turn(self, question: str,
                 answer: str, steps: int):
        """Record a complete question-answer turn."""

        turn = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "steps": steps
        }

        # Add to short-term memory
        self.short_term.append(turn)

        # Keep only recent turns
        if len(self.short_term) > self.max_short_term:
            self.short_term.pop(0)

        # Add to full log
        self.session_log.append(turn)

    def add_fact(self, fact: str, source: str = ""):
        """
        Store an important fact for long-term memory.
        Call this when the agent discovers something
        important that might be useful later.
        """
        self.long_term.append({
            "fact": fact,
            "source": source,
            "timestamp": datetime.now().isoformat()
        })

    def get_context_string(self) -> str:
        """
        Format recent conversation as context string.
        This gets injected into the agent's prompt
        so it knows what was discussed before.
        """
        if not self.short_term:
            return ""

        lines = ["PREVIOUS CONVERSATION:"]
        for turn in self.short_term[-3:]:  # last 3 turns
            lines.append(f"Q: {turn['question']}")
            # Truncate long answers
            answer = turn["answer"]
            if len(answer) > 200:
                answer = answer[:200] + "..."
            lines.append(f"A: {answer}")
            lines.append("")

        return "\n".join(lines)

    def get_facts_string(self) -> str:
        """Format long-term facts as context string."""
        if not self.long_term:
            return ""

        lines = ["KNOWN FACTS FROM THIS SESSION:"]
        for item in self.long_term[-5:]:  # last 5 facts
            lines.append(f"- {item['fact']}")

        return "\n".join(lines)

    def get_full_context(self) -> str:
        """Combine short-term and long-term memory."""
        parts = []

        facts = self.get_facts_string()
        if facts:
            parts.append(facts)

        context = self.get_context_string()
        if context:
            parts.append(context)

        return "\n\n".join(parts)

    def clear(self):
        """Clear short-term memory. Keep long-term."""
        self.short_term = []

    def save_session(self, path: str = "session_log.json"):
        """Save full session to disk."""
        with open(path, "w") as f:
            json.dump({
                "session_log": self.session_log,
                "long_term_facts": self.long_term
            }, f, indent=2)
        print(f"Session saved to {path}")

    def summary(self) -> dict:
        """Quick summary of memory state."""
        return {
            "short_term_turns": len(self.short_term),
            "long_term_facts": len(self.long_term),
            "total_turns": len(self.session_log)
        }


# ---- Test memory ----
if __name__ == "__main__":
    memory = ConversationMemory(max_short_term=5)

    # Simulate a conversation
    memory.add_turn(
        question="What is Shinto?",
        answer="Shinto is an ethnic religion focusing on "
               "ceremonies and rituals. Followers believe "
               "kami spirits are present in nature.",
        steps=2
    )

    memory.add_fact(
        "Japan practices Buddhism (66.7%) and Shinto (25.6%)",
        source="japan_culture.pdf"
    )

    memory.add_turn(
        question="How does Shinto connect to nature?",
        answer="Shinto connects to nature through kami "
               "spirits which are believed to inhabit "
               "rocks, trees, and mountains.",
        steps=3
    )

    memory.add_fact(
        "Kami spirits in Shinto are present in nature",
        source="japan_culture.pdf"
    )

    # Show what the agent would see
    print("=== FULL CONTEXT THE AGENT SEES ===")
    print(memory.get_full_context())

    print("\n=== MEMORY SUMMARY ===")
    print(memory.summary())

    # Test truncation — add more than max_short_term
    print("\n=== TESTING MEMORY LIMIT ===")
    for i in range(6):
        memory.add_turn(
            question=f"Question {i}",
            answer=f"Answer {i}",
            steps=1
        )

    print(f"Short term turns kept: "
          f"{len(memory.short_term)} "
          f"(max is 5)")

    memory.save_session()