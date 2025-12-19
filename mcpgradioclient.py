"""
MCP Gradio Client for Sentiment Analysis

This module creates a chat interface that connects to an MCP (Model Context Protocol) server
hosting sentiment analysis tools. It uses the smolagents library to create an AI agent
that can access and utilize the sentiment analysis capabilities exposed via MCP.

The client connects to a Gradio-hosted MCP server using Server-Sent Events (SSE) transport
and provides a conversational interface for users to interact with sentiment analysis tools.
"""

import gradio as gr
import os
import sys

from smolagents import InferenceClientModel, CodeAgent, MCPClient

# Initialize MCP client as None for proper cleanup in finally block
mcp_client = None

try:
    # Create the MCP client to connect to the sentiment analysis server
    # Uses SSE (Server-Sent Events) transport for real-time communication
    # Note: Don't pass structured_output to transports that don't accept it
    mcp_client = MCPClient(
        {"url": "https://joabyssinia-mcp-sentiment.hf.space/gradio_api/mcp/sse", "transport": "sse"}

    )

    # Retrieve available tools from the MCP server (sentiment analysis functions)
    tools = mcp_client.get_tools()
    
    # Initialize the language model using HuggingFace's Inference API
    # Requires HUGGINGFACE_API_TOKEN environment variable to be set
    model = InferenceClientModel(token=os.getenv("HUGGINGFACE_API_TOKEN"))
    
    # Create a CodeAgent that can execute code and use the MCP tools
    # Additional imports are authorized for the agent to safely handle JSON, AST parsing, URLs, and base64 encoding
    agent = CodeAgent(tools=[*tools], model=model, additional_authorized_imports=["json", "ast", "urllib", "base64"])

    # Create a chat interface where users can interact with the agent
    # The agent processes each message and returns responses using the sentiment analysis tools
    demo = gr.ChatInterface(
        fn=lambda message, history: str(agent.run(message)),
        examples=["Analyze the sentiment of the following text 'This is awesome'"],
        title="Agent with MCP Tools",
        description="This is a simple agent that uses MCP tools to answer questions.",
    )

    # Launch the Gradio interface with public sharing enabled
    demo.launch(share=True)
finally:
    # Ensure proper cleanup of MCP client connection on exit
    mcp_client.disconnect()
