"""智招Agents - 招商引资智能辅助系统 Streamlit UI"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from openai import OpenAI
from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL


@st.cache_resource
def get_rag():
    from src.rag_engine import RAGEngine
    return RAGEngine()


@st.cache_resource
def get_enterprise_tool():
    from src.enterprise_tool import EnterpriseTool
    return EnterpriseTool()


@st.cache_resource
def get_llm_client():
    return OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)


def page_rag_query():
    """智能信息查询页面"""
    st.header("智能信息查询")
    st.markdown("查询企业概况、匹配产业政策——基于本地知识库的RAG检索")

    rag = get_rag()
    ent_tool = get_enterprise_tool()
    client = get_llm_client()

    # 快捷查询按钮
    quick = st.radio("查询类型", ["企业查询", "政策检索", "自由提问"], horizontal=True)

    if quick == "企业查询":
        enterprises = ent_tool.list_all()
        options = [e["name"] for e in enterprises]
        selected = st.selectbox("选择目标企业", options)
        question = f"请介绍{selected}的详细情况，包括主营业务、投资动态和核心关注点"
    elif quick == "政策检索":
        policy_tags = ["新能源汽车", "储能", "人工智能", "低空经济", "总部经济", "制造业", "税收优惠", "人才政策", "用地保障"]
        tag = st.selectbox("选择政策方向", policy_tags)
        question = f"关于{tag}方面有什么优惠政策和扶持措施？"
    else:
        question = st.text_area("输入你的问题", placeholder="例如：中山市对新能源企业有什么优惠政策？比亚迪在华南有什么扩张计划？")

    if st.button("查询", type="primary", key="btn_rag"):
        if not question:
            st.warning("请输入问题")
            return

        with st.spinner("正在检索..."):
            # 1. RAG检索相关文档
            docs = rag.search(question, top_k=5)
            if not docs:
                st.warning("未找到相关信息")
                return

            context = "\n\n---\n\n".join([d["content"] for d in docs])
            sources = list(set([d["metadata"].get("source", "未知") for d in docs]))

            # 2. LLM生成回答
            prompt = f"""你是一位专业的招商顾问助手。根据以下知识库内容回答用户问题。
如果知识库中有相关信息，请依据知识库内容给出详细回答。
如果知识库中没有直接相关信息，请基于你的知识给出合理建议。

## 知识库内容
{context}

## 用户问题
{question}

请用中文回答，结构清晰，重点突出。如涉及具体金额或比例，请准确引用。"""

            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000,
            )
            answer = resp.choices[0].message.content

        # 显示结果
        st.subheader("查询结果")
        st.markdown(answer)

        with st.expander(f"参考来源 ({len(sources)}条)"):
            for s in sources:
                st.caption(f"- {s}")
            st.divider()
            for d in docs:
                st.caption(d["content"][:300] + "...")


def page_negotiation():
    """模拟谈判对话页面"""
    st.header("模拟谈判演练")
    st.markdown("AI扮演企业负责人，与招商人员进行多轮投资谈判模拟")

    from src.negotiation_agent import NegotiationAgent

    # 初始化
    if "nego_agent" not in st.session_state:
        st.session_state.nego_agent = None
        st.session_state.nego_messages = []
        st.session_state.nego_started = False

    # 第一步：选择对象企业
    if not st.session_state.nego_started:
        ent_tool = get_enterprise_tool()
        enterprises = ent_tool.list_all()
        col1, col2 = st.columns([3, 1])
        with col1:
            selected = st.selectbox("选择谈判对象企业", [e["name"] for e in enterprises])
        with col2:
            st.write("")
            st.write("")
            if st.button("开始谈判", type="primary"):
                with st.spinner("企业负责人准备中..."):
                    agent = NegotiationAgent()
                    opening = agent.start(selected)
                    st.session_state.nego_agent = agent
                    st.session_state.nego_messages = [
                        {"role": "assistant", "content": opening}
                    ]
                    st.session_state.nego_started = True
                st.rerun()

        # 显示企业画像
        profile = ent_tool.get_profile_text(selected)
        with st.expander("查看企业画像"):
            st.text(profile)

    # 第二步：多轮对话
    else:
        # 显示对话历史
        for msg in st.session_state.nego_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # 招商人员输入
        if prompt := st.chat_input("输入你的谈判话术..."):
            st.session_state.nego_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.spinner("企业负责人思考中..."):
                reply = st.session_state.nego_agent.respond(prompt)
            st.session_state.nego_messages.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)

        # 结束按钮
        if st.button("结束本轮谈判"):
            st.session_state.nego_agent = None
            st.session_state.nego_messages = []
            st.session_state.nego_started = False
            st.rerun()


def main():
    st.set_page_config(
        page_title="智招Agents - 招商智能辅助系统",
        page_icon="🏢",
        layout="wide",
    )
    st.title("🏢 智招Agents —— 招商引资智能辅助系统")
    st.caption("基于LangChain + Chroma + DeepSeek | 数据本地化 | 模拟谈判演练")

    tab1, tab2 = st.tabs(["智能信息查询", "模拟谈判演练"])
    with tab1:
        page_rag_query()
    with tab2:
        page_negotiation()


if __name__ == "__main__":
    main()
