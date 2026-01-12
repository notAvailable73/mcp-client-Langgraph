
# LangGraph Agent with MCP
This project is designed to efficiently integrate Model Context Protocol (MCP) with a LangGraph Agent, allowing it to dynamically access external tools, data sources, and APIs.

Using this project, you can connect a LangGraph agent to MCP servers and use predefined tools to perform various tasks, such as web searches and summarizing YouTube videos. You can also add additional servers as needed.

This integration enables automatic tool discovery and multi-server support, making AI systems more modular and powerful. It allows AI systems to automatically find tools and connect to multiple servers, increasing their flexibility and efficiency.

![image](https://github.com/user-attachments/assets/3540530a-7eed-4a47-87cf-00ee46fa5024)


## What is MCP and Why It Matters?

The Model Context Protocol (MCP) is an open standard that provides a structured way for AI applications to interact with external data, tools, and APIs. MCP was developed by Anthropic to address the challenge of dynamically connecting LLMs to external data sources without requiring custom integrations for each tool.

MCP is important because it helps AI systems share and access data easily, removing barriers between different tools. This makes AI more connected and efficient. It also allows developers to build smarter AI systems that can work with many different tools and grow easily.

## Project Structure

```bash
langgraph_mcp/
│-- agent.py  
│-- servers/
│   ├── tavily.py          
│   ├── yt_transcript.py   
│   ├── math_server.py 
│   ├──weather.py
│-- .env          
│-- requirements.txt   

```

## Installation

1. Clone the Repository:
```bash
git clone https://github.com/your-repo/langgraph-mcp.git
cd langgraph-mcp
```

2. Install Dependencies
```bash
pip install -r requirements.txt
```
3. Create a .env file and add:
```bash
TAVILY_API_KEY=<your-tavily-api-key>
OPENAI_API_KEY=<your-openai-api-key>
```

## How It Works
To start, run ```servers/server.py``` in your terminal. This will start the MCP server. Then, in a new terminal, run ```agent.py```. The agent will connect to the server via the MCP client and execute your query, as shown in the demo.

https://github.com/user-attachments/assets/c65255a1-62af-4711-8f10-ebfdacd9b3c6
