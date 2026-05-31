import datetime
from core.state import AgentState, AgentLog
from core.llm import get_llm
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

PLANNER_SYSTEM_PROMPT = """You are a Senior GIS (Geographic Information Systems) Architect. 
Your role is to analyze the user's spatial problem or request, and design a detailed, step-by-step geospatial analysis plan.

Your plan must address the following critical GIS concerns:
1. Coordinate Reference Systems (CRS): Specify which CRS/EPSG codes to use (e.g., EPSG:4326 for unprojected latitude/longitude, EPSG:3857 for web tiles, or specific projected systems like UTM zones for accurate metric measurements such as buffering and area calculations).
2. Required Libraries: Specify which Python libraries to use (e.g., `geopandas`, `shapely`, `folium`, `fiona`, `pyproj`, `rasterio`, `matplotlib`).
3. Spatial Methodology: Outline the exact geospatial operations needed (e.g., spatial joins, centroid calculations, spatial indexing (R-tree), buffering, intersection, data conversions).
4. Output Details: Design how the results should be saved, logged, or visualized (e.g., HTML interactive maps, GeoJSON, static PNG maps).

Do NOT write code. Provide only the architectural plan, rationale, and steps.
Format your output clearly using markdown.
"""

def planner_node(state: AgentState, config: RunnableConfig = None) -> dict:
    """
    GIS Planner agent node. Analyzes the spatial request and produces a step-by-step plan.
    """
    configurable = config.get("configurable", {}) if config else {}
    provider = configurable.get("provider", None)
    model_name = configurable.get("model_name", None)
    temperature = configurable.get("temperature", None)
    api_key = configurable.get("api_key", None)
    base_url = configurable.get("base_url", None)
    
    # Instantiate LLM
    llm = get_llm(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url
    )
    
    task = state["task"]
    
    # Construct prompts
    messages = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=f"Here is the geospatial task: {task}\n\nPlease draft the GIS execution plan.")
    ]
    
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    state_logs = state.get("logs", [])
    
    # Add a start log
    state_logs.append(AgentLog(
        agent="GIS Planner",
        log_type="thought",
        content="Analyzing the geospatial problem and planning the methodology...",
        timestamp=timestamp
    ))
    
    # Run the LLM
    try:
        response = llm.invoke(messages)
        plan_content = response.content
        
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        state_logs.append(AgentLog(
            agent="GIS Planner",
            log_type="output",
            content=plan_content,
            timestamp=timestamp
        ))
        
        return {
            "plan": plan_content,
            "logs": state_logs,
            "iterations": state.get("iterations", 0) + 1
        }
    except Exception as e:
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        state_logs.append(AgentLog(
            agent="GIS Planner",
            log_type="error",
            content=f"Failed to generate plan: {str(e)}",
            timestamp=timestamp
        ))
        return {
            "logs": state_logs,
            "plan": "Error occurred during planning.",
            "iterations": state.get("iterations", 0) + 1
        }
