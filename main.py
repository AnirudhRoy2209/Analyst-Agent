from agents.data_agent import data_agent
from langchain_core.messages import HumanMessage


def get_text(msg):
    """Normalize AIMessage.content (string or list of content blocks) to plain text."""
    content = msg.content
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "") if isinstance(b, dict) else str(b)
            for b in content
        )
    return content


if __name__ == "__main__":
    print("Analyst Agent ready. Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye.")
            break

        print("Thinking...", flush=True)          # <-- the recommendation goes here

        try:
            response = data_agent.invoke(
                {
                    "messages": [HumanMessage(content=user_input)],
                    "route_response": "",
                }
            )
            final_message = response["messages"][-1]
            print(f"\nAgent: {get_text(final_message)}\n")
        except Exception as e:
            print(f"\n[Error] {e}\n")