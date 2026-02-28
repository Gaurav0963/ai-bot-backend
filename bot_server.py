import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

app = Flask(__name__)
# Allow requests from your local index.html frontend
CORS(app) 

# GROQ API key is in render env variables and stored in my mac notes. (for me, in case i forget)

# 1. Define the LangGraph State Schema
class State(TypedDict):
    messages: Annotated[list, add_messages]

# 2. Initialize the LLM (Using Groq for fast inference)
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)

# Dynamic Lab Contexts mapping to the React Frontend's active tabs
LAB_CONTEXTS = {
    "labs": "Topic: Course Labs. Guide the student on solving AI problems without giving direct code. Focus heavily on BFS, DFS, and heuristic search methodologies like A*.",
    "playground": "Topic: ML Playground. The user is currently on the visualizer page. Discuss TensorFlow.js, neural networks, linear regression, and optimization (Adam) concepts.",
    "solutions": "Topic: Solution Vault. Explain the provided solutions conceptually and theoretically.",
    "resources": "Topic: External Resources. Recommend books, tutorials, or documentation related to AI.",
    "home": "Topic: General AI & Data Science concepts, PyTorch, and Python programming."
}

# 3. Define the Node Function
def chatbot_node(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

# 4. Build the LangGraph
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot_node)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# Add Memory tracking
memory = MemorySaver()
app_graph = graph_builder.compile(checkpointer=memory)

@app.route('/', methods=['GET'])
def health_check():
    return "IIIT AI Bot Server is running and ready! 🚀"

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_msg = data.get('message', '')
    
    # We dynamically fetch the context based on where the student is in the React app
    lab_id = data.get('lab_id', 'home') 
    thread_id = data.get('thread_id', 'new_thread')
    file_name = data.get('file_name', '')
    file_content = data.get('file_content', '')

    # Ensure we grab the right context, defaulting to home if unknown
    active_context = LAB_CONTEXTS.get(lab_id, LAB_CONTEXTS['home'])

    # The "Cheat Guard" Persona
    system_prompt = f"""You are an elite AI Teaching Assistant for AI & Data Science students.
    STRICT RULES:
    1. NEVER write the complete final code.
    2. Refuse requests for direct answers or completed assignments.
    3. Point out specific bugs in attached code, explain concepts logically, and guide students to the next step.
    4. Keep your tone encouraging but academically rigorous.
    
    Current Tab/Context the student is viewing: {active_context}
    """

    # Assemble the user's prompt, injecting the file if it exists
    full_user_msg = user_msg
    if file_content:
        full_user_msg += f"\n\n[Attached File Context: {file_name}]\n```python\n{file_content}\n```"

    # Configure the thread memory
    config = {"configurable": {"thread_id": thread_id}}

    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=full_user_msg)
        ]
        
        # Invoke the graph with history
        output_state = app_graph.invoke({"messages": messages}, config)
        
        # Extract the bot's latest reply
        bot_reply = output_state["messages"][-1].content
        
        return jsonify({
            "reply": bot_reply, 
            "thread_id": thread_id 
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Failed to connect to the AI model. Check backend console logs."}), 500

if __name__ == '__main__':
    # Running on 5001 to match the fetch request in index.html
    app.run(port=5001, debug=True)
