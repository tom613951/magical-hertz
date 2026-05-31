import streamlit as st
import datetime
import time
from core.state import AgentState, AgentLog
from core.config import Config
from agents.graph import create_gis_graph

# Setup page layout
st.set_page_config(
    page_title="GeoGraph - GIS Multi-Agent Collaboration System",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS that adapts to both Light and Dark themes
st.markdown("""
<style>
    /* Styling for agent timeline cards */
    .agent-card {
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 16px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        background-color: rgba(128, 128, 128, 0.05);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.02);
    }
    .agent-header {
        font-weight: 700;
        font-size: 1.15em;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .agent-badge {
        font-size: 0.75em;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
        color: white;
    }
    .badge-planner { background-color: #3b82f6; }
    .badge-developer { background-color: #10b981; }
    .badge-qa { background-color: #f59e0b; }
    .badge-system { background-color: #6b7280; }
    .badge-error { background-color: #ef4444; }
    
    .log-time {
        font-size: 0.85em;
        opacity: 0.7;
    }
    .log-content {
        margin-top: 10px;
        line-height: 1.6;
    }
    /* Hide top Streamlit elements for premium feel */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Main Application Title
st.title("🌍 GeoGraph")
st.caption("A LangGraph-powered Multi-Agent team specializing in Geospatial Analysis & Python GIS automation")

# Sidebar - Settings Panel
with st.sidebar:
    st.header("⚙️ System Configuration")
    
    # LLM Provider Selection
    provider = st.selectbox(
        "LLM Provider",
        options=["openai", "deepseek", "gemini", "anthropic", "ollama"],
        index=["openai", "deepseek", "gemini", "anthropic", "ollama"].index(Config.DEFAULT_PROVIDER)
    )
    
    # Dynamic settings based on selected provider
    api_key_default = ""
    base_url_default = ""
    
    if provider == "openai":
        api_key_default = Config.OPENAI_API_KEY
        base_url_default = Config.OPENAI_API_BASE
    elif provider == "deepseek":
        api_key_default = Config.DEEPSEEK_API_KEY
        base_url_default = Config.DEEPSEEK_API_BASE
    elif provider == "gemini":
        api_key_default = Config.GOOGLE_API_KEY
    elif provider == "anthropic":
        api_key_default = Config.ANTHROPIC_API_KEY
        
    api_key = ""
    if provider != "ollama":
        api_key = st.text_input(
            f"{provider.capitalize()} API Key",
            value=api_key_default,
            type="password",
            help="If configured in .env, it loads automatically."
        )
        
    base_url = ""
    if provider in ["openai", "deepseek", "ollama"]:
        base_url_help = "Ollama Host URL" if provider == "ollama" else "API Base Endpoint URL"
        base_url_val = Config.OLLAMA_HOST if provider == "ollama" else base_url_default
        base_url = st.text_input(
            "API Base / Host URL",
            value=base_url_val,
            help=base_url_help
        )
        
    # Model Selection
    default_model = Config.get_default_model(provider)
    model_name = st.text_input("Model Name", value=default_model)
    
    # Temperature and Iteration sliders
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=Config.DEFAULT_TEMPERATURE, step=0.1)
    max_iter = st.slider("Max Revision Loops", min_value=1, max_value=5, value=3)

# Sample tasks for quick input
sample_tasks = [
    "Load point data, project it to metric CRS, buffer by 500m, perform spatial join with polygon data, and plot on a Folium map.",
    "Calculate the shortest path between two points on a road network using NetworkX and geopandas, and export the route as GeoJSON.",
    "Read a raster elevation model (DEM) using rasterio, compute slope, slice slopes greater than 15 degrees, and plot the result static map."
]

st.subheader("🚀 Define Geospatial Task")
task_selection = st.selectbox("Choose a sample task template:", ["Custom Task"] + sample_tasks)

if task_selection == "Custom Task":
    task_input = st.text_area("Or type your custom GIS requirement here:", value="", height=120)
else:
    task_input = st.text_area("Your GIS requirement:", value=task_selection, height=120)

# Build run workflow section
if st.button("Start Agent Collaboration", type="primary", use_container_width=True):
    if not task_input.strip():
        st.warning("Please specify a geospatial task first!")
    elif provider != "ollama" and not api_key:
        st.error(f"API key is required for provider '{provider}'. Please fill it in the sidebar.")
    else:
        st.info("Compiling StateGraph and starting collaboration...")
        
        # 1. Initialize empty containers for visual progress
        status_container = st.empty()
        
        # We will use Tabs to organize the logs, final code, etc.
        tab_timeline, tab_code, tab_plan, tab_qa = st.tabs([
            "💬 Team Chat & Timeline", 
            "💻 Generated GIS Script", 
            "📋 Spatial Design Plan", 
            "🔍 QA Audit Reports"
        ])
        
        # Prepare LangGraph compilation
        graph = create_gis_graph()
        
        initial_state = AgentState(
            task=task_input,
            plan="",
            draft_code="",
            explanation="",
            qa_feedback="",
            qa_approved=False,
            iterations=0,
            max_iterations=max_iter,
            logs=[]
        )
        
        # Prepare runtime config overrides
        graph_config = {
            "configurable": {
                "provider": provider,
                "model_name": model_name,
                "temperature": temperature,
                "api_key": api_key if provider != "ollama" else None,
                "base_url": base_url if provider in ["openai", "deepseek", "ollama"] else None
            }
        }
        
        # Run workflow in a spinner block
        with st.spinner("GeoGraph team is working..."):
            try:
                # Invoke the LangGraph execution
                final_state = graph.invoke(initial_state, config=graph_config)
                
                # Report Status
                if final_state.get("qa_approved", False):
                    status_container.success("🎉 GeoGraph Completed! GIS script generated and approved by QA Inspector.")
                else:
                    status_container.warning(f"⚠️ GeoGraph completed but without QA Approval or reached revision limit ({final_state.get('iterations', 0)} loops).")
                
                # Render Timeline Logs
                with tab_timeline:
                    st.write("### Collaboration Execution Log")
                    for log in final_state.get("logs", []):
                        agent = log.get("agent")
                        log_type = log.get("log_type", "info")
                        content = log.get("content", "")
                        time_str = log.get("timestamp", "")
                        
                        # Set badge class based on agent type
                        badge_class = "badge-system"
                        if "Planner" in agent:
                            badge_class = "badge-planner"
                        elif "Developer" in agent:
                            badge_class = "badge-developer"
                        elif "QA" in agent:
                            badge_class = "badge-qa"
                        elif log_type == "error":
                            badge_class = "badge-error"
                            
                        st.markdown(f"""
                        <div class="agent-card">
                            <div class="agent-header">
                                <span>
                                    <span class="agent-badge {badge_class}">{agent}</span>
                                    &nbsp;({log_type.upper()})
                                </span>
                                <span class="log-time">🕒 {time_str}</span>
                            </div>
                            <div class="log-content">
                                {content.replace(chr(10), '<br>')}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Render Generated Code Tab
                with tab_code:
                    code = final_state.get("draft_code", "")
                    explanation = final_state.get("explanation", "")
                    
                    if code:
                        st.write("### Final GIS Automation Script (`gis_analysis.py`)")
                        st.code(code, language="python")
                        st.download_button(
                            label="📥 Download Python GIS Script",
                            data=code,
                            file_name="gis_analysis.py",
                            mime="text/x-python"
                        )
                        st.write("### Developer Explanation")
                        st.write(explanation)
                    else:
                        st.info("No code has been successfully generated yet.")
                
                # Render Spatial Design Plan Tab
                with tab_plan:
                    st.write("### GIS Spatial Methodology Plan")
                    st.markdown(final_state.get("plan", "No plan created."))
                
                # Render QA Audit Reports
                with tab_qa:
                    st.write("### GIS QA Auditing History & Revision Loop Details")
                    st.write(f"**Total revision iterations**: {final_state.get('iterations', 0)}")
                    st.markdown(final_state.get("qa_feedback", "No QA audits have run."))
                    
            except Exception as e:
                status_container.error(f"Failed to execute GeoGraph workflow: {str(e)}")
                st.exception(e)
