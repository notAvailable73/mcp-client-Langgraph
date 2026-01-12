import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
load_dotenv()

# Example queries:
# "Get me the full member list of my organization"
# "What is Agentic AI?"
# "What is today's date?"

# Define custom tool


# Define llm
model = ChatOpenAI(model="gpt-4o")
checkpointer = InMemorySaver()

# Define MCP servers
async def run_agent():
    client = MultiServerMCPClient(
        {
            "tavily": {
                "command": "python",
                "args": ["servers/tavily.py"],
                "transport": "stdio",
            },
            "youtube_transcript": {
                "command": "python",
                "args": ["servers/yt_transcript.py"],
                "transport": "stdio",
            },
            "dateTime": {
                "command": "python",
                "args": ["servers/dateTime.py"],
                "transport": "stdio",
            },
            "math": {
                "command": "python",
                "args": ["servers/math.py"],
                "transport": "stdio",
            },
            "clickup": {
                "command": "npx",
                "args": ["-y", "mcp-remote", "https://mcp.clickup.com/mcp"],
                "transport": "stdio",
            },
            # "weather": {
            # "url": "http://localhost:8000/sse", # start your weather server on port 8000
            # "transport": "sse",
            # }
        }
    )

    mcp_tools = await client.get_tools()

    # Combine MCP tools with custom tools
    tools = mcp_tools + [get_present_date]

    # Bind tools to the model
    model_with_tools = model.bind_tools(tools)

    # Define the function that calls the model
    def call_model(state: MessagesState):
        messages = state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

    # Define the conditional edge function
    def should_continue(state: MessagesState):
        messages = state["messages"]
        last_message = messages[-1]
        # If there are tool calls, continue to tools node
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        # Otherwise, end
        return END

    # Create the graph
    workflow = StateGraph(MessagesState)

    # Add nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))

    # Add edges
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, ["tools", END])
    workflow.add_edge("tools", "agent")

    # Compile the graph with checkpointer
    agent = workflow.compile(checkpointer=checkpointer)

    # System message
    system_message = SystemMessage(content=(
        "You have access to multiple tools that can help answer queries. "
        "Use them dynamically and efficiently based on the user's request. "
    ))

    # Thread ID for conversation continuity
    thread_id = "conversation-1"
    config = {"configurable": {"thread_id": thread_id}}

    # Initialize conversation with system message
    messages = [system_message]

    print("Multi-turn Agent Chat (type 'quit', 'exit', or 'bye' to end)")
    print("-" * 60)

    # Conversation loop
    while True:
        query = input("\nYou: ").strip()

        # Exit conditions
        if query.lower() in ['quit', 'exit', 'bye', 'q']:
            print("\nGoodbye!")
            break

        # Skip empty queries
        if not query:
            continue

        # Add user message to conversation
        messages.append(HumanMessage(content=query))

        # Process the query with conversation history
        agent_response = await agent.ainvoke(
            {"messages": messages},
            config=config
        )

        # Update messages with full conversation state
        messages = agent_response["messages"]

        # Print assistant response
        response_content = agent_response["messages"][-1].content
        print(f"\nAssistant: {response_content}")

# Run the agent
if __name__ == "__main__":
    asyncio.run(run_agent())