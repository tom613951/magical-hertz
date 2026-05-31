import datetime
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
from core.state import AgentState, AgentLog
from agents.planner import planner_node
from agents.developer import developer_node
from agents.qa import qa_node
from tools.file_tool import write_workspace_file

# Node to save the final approved file to workspace
def saver_node(state: AgentState, config: RunnableConfig = None) -> dict:
    """
    Saves the QA-approved draft code to the local workspace.
    """
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    state_logs = state.get("logs", [])
    
    if not state.get("qa_approved", False):
        state_logs.append(AgentLog(
            agent="GeoGraph System",
            log_type="thought",
            content="Workflow ended without QA approval. Skipping file generation.",
            timestamp=timestamp
        ))
        return {"logs": state_logs}
        
    code = state.get("draft_code", "")
    if not code:
        state_logs.append(AgentLog(
            agent="GeoGraph System",
            log_type="error",
            content="Approved script is empty. Skipping file write.",
            timestamp=timestamp
        ))
        return {"logs": state_logs}
        
    # Write the file
    filename = "gis_analysis.py"
    write_res = write_workspace_file.invoke({"filename": filename, "content": code})
    
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    state_logs.append(AgentLog(
        agent="GeoGraph System",
        log_type="output",
        content=write_res,
        timestamp=timestamp
    ))
    
    return {"logs": state_logs}

def route_qa(state: AgentState) -> str:
    """
    Route based on QA approval and iteration count.
    """
    if state.get("qa_approved", False):
        return "save"
    
    iterations = state.get("iterations", 0)
    max_iterations = state.get("max_iterations", 3)
    
    if iterations >= max_iterations:
        return "save"  # Save whatever we have at the limit
        
    return "developer"

def create_gis_graph():
    """
    Compiles and returns the LangGraph StateGraph.
    """
    workflow = StateGraph(AgentState)
    
    # Register nodes
    from agents.planner import planner_node # Fixed import here
    workflow.add_node("planner", planner_node)
    workflow.add_node("developer", developer_node)
    workflow.add_node("qa", qa_node)
    workflow.add_node("save", saver_node)
    
    # Establish edges
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "developer")
    workflow.add_edge("developer", "qa")
    
    # Conditional routing after QA
    workflow.add_conditional_edges(
        "qa",
        route_qa,
        {
            "developer": "developer",
            "save": "save"
        }
    )
    
    workflow.add_edge("save", END)
    
    return workflow.compile()
