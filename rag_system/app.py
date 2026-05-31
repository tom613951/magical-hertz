import sys
import os

# Ensure the workspace root is in sys.path so we can import 'core.config' and 'core.llm'
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import streamlit as st
import io
from langchain_core.messages import SystemMessage, HumanMessage
from core.config import Config
from core.llm import get_llm
from rag_core.database import RAGDatabase
from rag_core.retrieval import AdvancedRetriever
from utils.parser import parse_file, chunk_text

# 设置页面配置
st.set_page_config(
    page_title="GeoGraph RAG - 高级智能文档问答",
    page_icon="📚",
    layout="wide"
)

# 自定义 CSS 样式（自适应深色/浅色模式）
st.markdown("""
<style>
    .chunk-card {
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 12px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        background-color: rgba(128, 128, 128, 0.03);
    }
    .score-badge {
        font-size: 0.78em;
        padding: 2px 6px;
        border-radius: 4px;
        color: white;
        font-weight: bold;
        margin-right: 5px;
    }
    .badge-rerank { background-color: #ec4899; }
    .badge-vector { background-color: #3b82f6; }
    .badge-bm25 { background-color: #10b981; }
    .badge-source { background-color: #6b7280; }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# 主页标题
st.title("📚 GeoGraph - 高级 RAG 知识库")
st.caption("融合大模型查询扩展、向量数据库与 BM25 混合检索，以及 Flashrank 本地模型重排的智能文档问答系统（自适应深浅色主题）")

# 侧边栏配置面板
with st.sidebar:
    st.header("⚙️ 系统配置")
    
    # 大模型服务商选择
    provider = st.selectbox(
        "大模型服务商 (LLM)",
        options=["openai", "deepseek", "gemini", "anthropic", "ollama"],
        index=["openai", "deepseek", "gemini", "anthropic", "ollama"].index(Config.DEFAULT_PROVIDER)
    )
    
    # 动态载入默认 API 密钥和代理地址
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
        base_url_val = Config.OLLAMA_HOST if provider == "ollama" else base_url_default
        base_url = st.text_input("API 代理地址 / Host 地址", value=base_url_val)
        
    # 模型选择
    default_model = Config.get_default_model(provider)
    model_name = st.text_input("模型名称 (Model)", value=default_model)
    temperature = st.slider("温度 (Temperature)", min_value=0.0, max_value=1.0, value=0.0, step=0.1) # 0.0 最适合事实问答
    
    st.markdown("---")
    st.header("🧹 向量数据库工具")
    
    # 清空向量库按钮
    if st.button("清空本地向量数据库", type="secondary", use_container_width=True):
        res = RAGDatabase.clear_database()
        st.success(res)
        st.rerun()

# 文档载入区域
st.subheader("📁 1. 上传文档至知识库")
uploaded_files = st.file_uploader(
    "选择要解析的 PDF, Markdown, TXT 或 JSON/GeoJSON 文档：",
    type=["pdf", "md", "txt", "json", "geojson"],
    accept_multiple_files=True
)

if uploaded_files:
    if st.button("构建/更新知识库", type="primary"):
        # 开始构建前校验密钥
        if provider != "ollama" and not api_key:
            st.error("运行向量化嵌入（embeddings）需要配置 API 密钥！")
        else:
            with st.spinner("正在解析上传的文档并计算生成向量表示..."):
                try:
                    db = RAGDatabase(provider=provider, api_key=api_key)
                    all_chunks = []
                    
                    for uploaded_file in uploaded_files:
                        filename = uploaded_file.name
                        file_bytes = io.BytesIO(uploaded_file.read())
                        
                        # 解析文本
                        raw_text = parse_file(file_bytes, filename)
                        
                        # 文本分块
                        chunks = chunk_text(raw_text, filename)
                        all_chunks.extend(chunks)
                    
                    if all_chunks:
                        # 存入向量数据库
                        db.add_documents(all_chunks)
                        st.success(f"成功解析 {len(uploaded_files)} 个文档，并将 {len(all_chunks)} 个文本切片导入 Chroma 向量数据库！")
                    else:
                        st.warning("无法从上传的文档中提取出有效的文本内容。")
                except Exception as e:
                    st.error(f"构建知识库失败: {str(e)}")

# 提问区域
st.subheader("💬 2. 问答与知识检索")
query_input = st.text_input("请输入您的问题：")

if st.button("检索知识库", use_container_width=True):
    if not query_input.strip():
        st.warning("请先输入您的问题！")
    elif provider != "ollama" and not api_key:
        st.error("开始提问前，请在左侧边栏配置您的 API 密钥。")
    else:
        with st.spinner("正在检索匹配知识切片并合成回答中..."):
            try:
                # 1. 初始化数据库与检索服务
                db = RAGDatabase(provider=provider, api_key=api_key)
                
                # 初始化 LLM
                llm = get_llm(
                    provider=provider,
                    model_name=model_name,
                    temperature=temperature,
                    api_key=api_key,
                    base_url=base_url if provider in ["openai", "deepseek", "ollama"] else None
                )
                
                retriever = AdvancedRetriever(db, llm)
                
                # 2. 混合检索并重排
                retrieved_results = retriever.retrieve(query_input, top_k=5)
                
                if not retrieved_results:
                    st.warning("数据库中未检索到相关文档切片。请先上传并构建您的知识库文档！")
                else:
                    # 3. 拼接检索上下文
                    context_blocks = []
                    for idx, res in enumerate(retrieved_results, 1):
                        doc = res["document"]
                        source = doc.metadata.get("source", "Unknown")
                        context_blocks.append(f"--- Context Segment {idx} (Source: {source}) ---\n{doc.page_content}")
                    
                    context_str = "\n\n".join(context_blocks)
                    
                    # 4. 基于大模型生成问答
                    system_prompt = """You are a highly analytical QA assistant. Answer the user's question precisely and truthfully based ONLY on the provided context segments. 
If the context does not contain the answer, reply: "I cannot find the answer in the provided documents." Do not invent facts.

Context:
{context}
"""
                    response = llm.invoke([
                        SystemMessage(content=system_prompt.format(context=context_str)),
                        HumanMessage(content=query_input)
                    ])
                    
                    # 5. 分 Tab 页展示结果
                    tab_ans, tab_source, tab_queries = st.tabs([
                        "✏️ 智能回答", 
                        "🔍 检索切片展示 (重排)", 
                        "⚙️ 联想检索词 (查询扩展)"
                    ])
                    
                    with tab_ans:
                        st.markdown("### 回答内容")
                        st.write(response.content)
                        
                    with tab_source:
                        st.markdown("### 最相关的召回切片 (由 Flashrank 重排)")
                        for idx, res in enumerate(retrieved_results, 1):
                            doc = res["document"]
                            source = doc.metadata.get("source", "Unknown")
                            chunk_id = doc.metadata.get("chunk_id", "?")
                            
                            st.markdown(f"""
                            <div class="chunk-card">
                                <div>
                                    <span class="score-badge badge-rerank">重排评分: {res['rerank_score']:.3f}</span>
                                    <span class="score-badge badge-vector">向量相似度: {res['vector_score']:.3f}</span>
                                    <span class="score-badge badge-bm25">BM25得分: {res['bm25_score']:.3f}</span>
                                    <span class="score-badge badge-source">源文档: {source} (切片 {chunk_id})</span>
                                </div>
                                <div style="margin-top: 10px; font-size: 0.95em;">
                                    {doc.page_content}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                    with tab_queries:
                        st.markdown("### 大模型为了提高召回率自动生成的联想检索词：")
                        from rag_core.retrieval import expand_query
                        expanded_queries = expand_query(query_input, llm)
                        for q in expanded_queries:
                            st.markdown(f"- *\"{q}\"*")
                            
            except Exception as e:
                st.error(f"问答检索执行失败: {str(e)}")
                st.exception(e)
