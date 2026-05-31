# Magical Hertz 🌍

基于 LangChain、LangGraph 与 Streamlit 构建的高端 GIS 多智能体协作系统与高级 RAG（检索增强生成）知识库。

---

## 🌟 核心模块

### 1. GeoGraph (GIS 多智能体协同系统)
基于状态图（StateGraph）构建的虚拟专家团队，模拟 GIS 规划师、开发工程师和 QA 质检员协同解决空间地理问题并编写 Python 自动化代码。
* **本地启动**: `streamlit run app.py`

### 2. Advanced RAG (高级地理信息知识库)
集成“查询扩展（LLM）”、“混合检索（Chroma 向量检索 + BM25 词频检索）”与“Flashrank 本地模型重排”的智能文档问答检索系统。
* **本地启动**: `streamlit run rag_system/app.py`

---

## 🚀 快速开始

1. **安装依赖环境**：
   ```bash
   pip install -r requirements.txt
   ```
2. **配置 API 密钥**：
   复制 `.env.example` 并重命名为 `.env`，填入对应大模型服务商的 API Key（支持 OpenAI、DeepSeek、Gemini、Anthropic 或本地运行的 Ollama）。
