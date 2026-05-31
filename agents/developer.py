import datetime
import re
from core.state import AgentState, AgentLog
from core.llm import get_llm
from tools.search_tool import web_search
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.runnables import RunnableConfig

DEVELOPER_SYSTEM_PROMPT = """You are a Senior GIS Developer and Spatial Analyst. 
Your task is to write a Python script that implements the GIS Planner's design to solve the user's spatial problem.

Guidelines for writing GIS code:
1. Libraries: Rely on standard GIS libraries like `geopandas`, `shapely`, `folium`, `pyproj`, `rasterio`, `matplotlib`.
2. Projections (CRS):
   - Always be mindful of projections. Coordinate distance calculations (like `.distance()`), buffer operations (like `.buffer()`), and area calculations (like `.area`) MUST be done on projected coordinate reference systems (e.g. UTM zones or EPSG:3857, metric units), NOT on geographic coordinates (EPSG:4326, degrees).
   - If visualizing with `folium` (which expects EPSG:4326 latitude/longitude), you MUST reproject your final layer back to EPSG:4326 using `.to_crs(epsg=4326)`.
3. Code Quality:
   - Provide clean, robust, and commented code.
   - Include imports at the beginning.
   - Use mock data generation, file inputs, or standard GeoJSON endpoints if real spatial data is needed.
   - Print clear debug statements to show step-by-step progress.

You have access to a `web_search` tool. If you are unsure of the API syntax for a specific version of geopandas, shapely, or folium, use the tool to lookup documentation.

Your final output MUST contain:
1. The full Python script enclosed in a single ```python ... ``` code block.
2. A brief, bulleted explanation of how the code works and any library dependencies.
"""

def developer_node(state: AgentState, config: RunnableConfig = None) -> dict:
    """
    GIS Developer agent node. Writes the Python GIS code based on the plan and QA feedback.
    Can perform web searches to resolve syntax issues.
    """
    configurable = config.get("configurable", {}) if config else {}
    provider = configurable.get("provider", None)
    model_name = configurable.get("model_name", None)
    temperature = configurable.get("temperature", None)
    api_key = configurable.get("api_key", None)
    base_url = configurable.get("base_url", None)
    
    llm = get_llm(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url
    )
    
    task = state["task"]
    plan = state["plan"]
    qa_feedback = state.get("qa_feedback", "")
    draft_code = state.get("draft_code", "")
    
    # Construct input prompt
    prompt_content = f"Task: {task}\n\nGIS Plan:\n{plan}\n"
    if qa_feedback:
        prompt_content += f"\nPrevious Draft Code:\n{draft_code}\n\nQA Inspector Feedback:\n{qa_feedback}\nPlease revise the code to resolve the issues reported by the QA Inspector."
        
    messages = [
        SystemMessage(content=DEVELOPER_SYSTEM_PROMPT),
        HumanMessage(content=prompt_content)
    ]
    
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    state_logs = state.get("logs", [])
    
    state_logs.append(AgentLog(
        agent="GIS Developer",
        log_type="thought",
        content="Developing python spatial script to implement the plan...",
        timestamp=timestamp
    ))
    
    # Try binding tools if model supports it
    try:
        llm_with_tools = llm.bind_tools([web_search])
    except Exception:
        llm_with_tools = llm # Fallback to standard llm if bind_tools fails
        
    # Agent execution loop for tool calls
    for i in range(3): # Max 3 tool cycles
        try:
            response = llm_with_tools.invoke(messages)
            messages.append(response)
            
            # Check for tool calls (native tool calling)
            tool_calls = getattr(response, "tool_calls", [])
            if not tool_calls:
                break
                
            # Execute tool calls
            for tc in tool_calls:
                if tc["name"] == "web_search":
                    query = tc["args"].get("query", "")
                    
                    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                    state_logs.append(AgentLog(
                        agent="GIS Developer",
                        log_type="tool_call",
                        content=f"Searching: {query}",
                        timestamp=timestamp
                    ))
                    
                    search_res = web_search.invoke({"query": query})
                    messages.append(ToolMessage(content=search_res, tool_call_id=tc["id"]))
                    
                    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                    state_logs.append(AgentLog(
                        agent="GIS Developer",
                        log_type="thought",
                        content=f"Found search results for '{query}'. Integrating details...",
                        timestamp=timestamp
                    ))
        except Exception as e:
            # Fallback execution in case tool call error occurs
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            state_logs.append(AgentLog(
                agent="GIS Developer",
                log_type="error",
                content=f"Error during tool-enabled run: {str(e)}. Attempting fallback run...",
                timestamp=timestamp
            ))
            response = llm.invoke(messages)
            break
            
    # Final response
    final_text = response.content
    
    # Parse code block
    code_match = re.search(r"```python\s*(.*?)\s*```", final_text, re.DOTALL)
    code = code_match.group(1) if code_match else ""
    
    # Parse explanation
    explanation = re.sub(r"```python\s*(.*?)\s*```", "", final_text, flags=re.DOTALL).strip()
    
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    state_logs.append(AgentLog(
        agent="GIS Developer",
        log_type="output",
        content=f"Draft Code Generated:\n{code[:300]}..." if code else "No code block found.",
        timestamp=timestamp
    ))
    
    return {
        "draft_code": code,
        "explanation": explanation,
        "logs": state_logs
    }
