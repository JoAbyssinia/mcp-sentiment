import gradio as gr
import os
import sys

from smolagents import InferenceClientModel, CodeAgent, MCPClient


mcp_client = None
try:
    # Create the MCP client (don't pass structured_output to transports that don't accept it)

    mcp_client = MCPClient(
        {"url": "https://joabyssinia-mcp-sentiment.hf.space/gradio_api/mcp/sse", "transport": "sse"}

    )

    tools = mcp_client.get_tools()
    model = InferenceClientModel(token=os.getenv("HUGGINGFACE_API_TOKEN"))
    agent = CodeAgent(tools=[*tools], model=model, additional_authorized_imports=["json", "ast", "urllib", "base64"])

    demo = gr.ChatInterface(
        fn=lambda message, history: str(agent.run(message)),
        examples=["Analyze the sentiment of the following text 'This is awesome'"],
        title="Agent with MCP Tools",
        description="This is a simple agent that uses MCP tools to answer questions.",
    )

    demo.launch(share=True)
finally:
     mcp_client.disconnect()
