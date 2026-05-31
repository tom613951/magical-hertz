import streamlit as st
import datetime
import time
from core.state import AgentState, AgentLog
from core.config import Config
from agents.graph import create_gis_graph

# 设置页面布局
st.set_page_config(
    page_title="GeoGraph - GIS多智能体协同系统",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义 CSS 样式（自动适配系统深色/浅色模式）
st.markdown("""
<style>
    /* 智能体卡片时间轴样式 */
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
    /* 隐藏 Streamlit 默认页眉/页脚以获得高端体验 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# 主应用标题
st.title("🌍 GeoGraph")
st.caption("基于 LangGraph 构建的地理空间分析与 Python GIS 自动化多智能体协同系统")

# 侧边栏 - 配置面板
with st.sidebar:
    st.header("⚙️ 系统配置")
    
    # 选择 LLM 服务商
    provider = st.selectbox(
        "大模型服务商 (LLM)",
        options=["openai", "deepseek", "gemini", "anthropic", "ollama"],
        index=["openai", "deepseek", "gemini", "anthropic", "ollama"].index(Config.DEFAULT_PROVIDER)
    )
    
    # 动态载入默认密钥
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
            f"{provider.capitalize()} API 密钥",
            value=api_key_default,
            type="password",
            help="如果在 .env 文件中配置了密钥，系统会自动加载。"
        )
        
    base_url = ""
    if provider in ["openai", "deepseek", "ollama"]:
        base_url_help = "Ollama 服务的 Host 地址" if provider == "ollama" else "API 代理或基础请求地址"
        base_url_val = Config.OLLAMA_HOST if provider == "ollama" else base_url_default
        base_url = st.text_input(
            "API 代理地址 / Host 地址",
            value=base_url_val,
            help=base_url_help
        )
        
    # 模型名称与参数调节
    model_name = st.text_input(
        "模型名称 (Model)", 
        value="", 
        placeholder="例如: gpt-4o, claude-3-5-sonnet, deepseek-reasoner",
        help="请在此输入你想使用的大模型名称"
    )
    temperature = st.slider("温度 (Temperature)", min_value=0.0, max_value=1.0, value=Config.DEFAULT_TEMPERATURE, step=0.1)
    max_iter = st.slider("最大质检修改循环数", min_value=1, max_value=5, value=3)

# 示例任务模版
sample_tasks = [
    "加载点数据，将其投影到投影坐标系（米），做500米缓冲区分析，与面数据进行空间连接，最后用 Folium 地图进行可视化展示。",
    "使用 NetworkX 和 GeoPandas 计算道路网络上两点之间的最短路径，并将路线导出为 GeoJSON 格式。",
    "使用 Rasterio 读取数字高程模型（DEM）栅格数据，计算坡度，筛选出坡度大于15度的区域，并绘制静态地图展示结果。"
]

st.subheader("🚀 定义地理空间任务")
task_selection = st.selectbox("选择一个示例任务模版:", ["自定义任务"] + sample_tasks)

if task_selection == "自定义任务":
    task_input = st.text_area("或者在此输入您的自定义 GIS 需求:", value="", height=120)
else:
    task_input = st.text_area("您的 GIS 需求:", value=task_selection, height=120)

# 启动智能体协同工作按钮
if st.button("启动智能体协同工作", type="primary", use_container_width=True):
    if not task_input.strip():
        st.warning("请先指定地理空间任务！")
    elif not model_name.strip():
        st.error("请输入模型名称 (Model)！")
    elif provider != "ollama" and not api_key:
        st.error(f"大模型服务商 '{provider}' 需要 API 密钥。请在左侧边栏中填写。")
    else:
        st.info("正在编译状态图并启动多智能体协作...")
        
        # 初始化界面提示与 Tabs 容器
        status_container = st.empty()
        
        tab_timeline, tab_code, tab_plan, tab_qa = st.tabs([
            "💬 团队对话与时间轴", 
            "💻 生成的 GIS 脚本", 
            "📋 空间分析设计方案", 
            "🔍 QA 质检报告"
        ])
        
        # 编译 LangGraph
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
        
        # 配置运行时参数
        graph_config = {
            "configurable": {
                "provider": provider,
                "model_name": model_name,
                "temperature": temperature,
                "api_key": api_key if provider != "ollama" else None,
                "base_url": base_url if provider in ["openai", "deepseek", "ollama"] else None
            }
        }
        
        # 执行工作流并渲染进度
        with st.spinner("GeoGraph 专家团队正在协同工作中..."):
            try:
                # 执行 LangGraph 编译的图
                final_state = graph.invoke(initial_state, config=graph_config)
                
                # 工作流状态反馈
                if final_state.get("qa_approved", False):
                    status_container.success("🎉 GeoGraph 协作完成！GIS 脚本已生成并通过 QA 质检工程师审核。")
                else:
                    status_container.warning(f"⚠️ GeoGraph 执行结束，但未获得 QA 审核通过或已达到修改次数上限（已循环迭代 {final_state.get('iterations', 0)} 次）。")
                
                # 渲染对话与时间轴
                with tab_timeline:
                    st.write("### 协作执行日志")
                    for log in final_state.get("logs", []):
                        agent = log.get("agent")
                        log_type = log.get("log_type", "info")
                        content = log.get("content", "")
                        time_str = log.get("timestamp", "")
                        
                        # 根据角色指定 badge 颜色
                        badge_class = "badge-system"
                        if "Planner" in agent or "规划" in agent:
                            badge_class = "badge-planner"
                        elif "Developer" in agent or "开发" in agent:
                            badge_class = "badge-developer"
                        elif "QA" in agent or "质检" in agent:
                            badge_class = "badge-qa"
                        elif log_type == "error":
                            badge_class = "badge-error"
                            
                        # 对英文角色进行中文显示映射优化
                        agent_zh = agent
                        if agent == "GIS Planner": agent_zh = "GIS 规划师"
                        elif agent == "GIS Developer": agent_zh = "GIS 开发工程师"
                        elif agent == "GIS QA Inspector": agent_zh = "GIS QA 质检员"
                        elif agent == "GeoGraph System": agent_zh = "系统核心"
                            
                        st.markdown(f"""
                        <div class="agent-card">
                            <div class="agent-header">
                                <span>
                                    <span class="agent-badge {badge_class}">{agent_zh}</span>
                                    &nbsp;({log_type.upper()})
                                </span>
                                <span class="log-time">🕒 {time_str}</span>
                            </div>
                            <div class="log-content">
                                {content.replace(chr(10), '<br>')}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # 渲染生成的代码
                with tab_code:
                    code = final_state.get("draft_code", "")
                    explanation = final_state.get("explanation", "")
                    
                    if code:
                        st.write("### 最终 GIS 自动化脚本 (`gis_analysis.py`)")
                        st.code(code, language="python")
                        st.download_button(
                            label="📥 下载 Python GIS 脚本",
                            data=code,
                            file_name="gis_analysis.py",
                            mime="text/x-python"
                        )
                        st.write("### 开发工程师说明")
                        st.write(explanation)
                    else:
                        st.info("尚未成功生成任何代码。")
                
                # 渲染设计方案
                with tab_plan:
                    st.write("### GIS 空间分析方案设计")
                    st.markdown(final_state.get("plan", "未创建任何方案。"))
                
                # 渲染质检报告
                with tab_qa:
                    st.write("### GIS QA 审核历史与循环修改详情")
                    st.write(f"**总修改迭代次数**: {final_state.get('iterations', 0)}")
                    st.markdown(final_state.get("qa_feedback", "未运行任何 QA 质检。"))
                    
            except Exception as e:
                status_container.error(f"多智能体工作流执行失败: {str(e)}")
                st.exception(e)
