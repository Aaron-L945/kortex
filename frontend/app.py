"""
Streamlit 前端

功能：
  - 登录界面（JWT）
  - 侧边栏：用户信息 + 文档管理（上传/删除/列表）
  - 主界面：对话框，支持流式输出
  - 显示来源引用（从 Agent 输出中解析）
"""

import json
import time

import requests
import streamlit as st

API_BASE = "http://localhost:8000/api"

st.set_page_config(
    page_title="企业知识库 Agent",
    page_icon="🏢",
    layout="wide",
)

# ─── Session State 初始化 ─────────────────────────────────────────────────────

if "token" not in st.session_state:
    st.session_state.token = None
if "user" not in st.session_state:
    st.session_state.user = None
if "messages" not in st.session_state:
    st.session_state.messages = []


# ─── 工具函数 ─────────────────────────────────────────────────────────────────

def api_headers():
    return {"Authorization": f"Bearer {st.session_state.token}"}


def login(username: str, password: str) -> bool:
    try:
        resp = requests.post(
            f"{API_BASE}/auth/token",
            data={"username": username, "password": password},
            timeout=10,
        )
        if resp.status_code == 200:
            st.session_state.token = resp.json()["access_token"]
            me = requests.get(f"{API_BASE}/auth/users/me", headers=api_headers()).json()
            st.session_state.user = me
            return True
    except Exception as e:
        st.error(f"连接服务器失败：{e}")
    return False


def logout():
    st.session_state.token = None
    st.session_state.user = None
    st.session_state.messages = []
    st.rerun()


def stream_chat(query: str):
    """调用非流式接口，一次性获取结果"""
    try:
        resp = requests.post(
            f"{API_BASE}/agent/chat",
            json={"message": query},
            headers=api_headers(),
            timeout=120,
        )
        if resp.status_code == 200:
            return resp.json()["response"]
        else:
            return f"[错误] {resp.status_code}: {resp.text}"
    except Exception as e:
        return f"[错误] 连接服务器失败：{e}"


def permission_badge(level: int) -> str:
    badges = {1: "🟢 公开", 2: "🟡 内部", 3: "🟠 机密", 4: "🔴 绝密"}
    return badges.get(level, str(level))


# ─── 登录页 ───────────────────────────────────────────────────────────────────

def render_login():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("## 🏢 企业知识库 Agent")
        st.markdown("---")
        with st.form("login_form"):
            username = st.text_input("用户名", placeholder="admin / hr_user / employee")
            password = st.text_input("密码", type="password", placeholder="密码")
            submitted = st.form_submit_button("登录", use_container_width=True)

        if submitted:
            if login(username, password):
                st.success("登录成功！")
                time.sleep(0.5)
                st.rerun()
            else:
                st.error("用户名或密码错误")

        st.markdown("""
**测试账号：**
| 用户名 | 密码 | 权限 |
|--------|------|------|
| admin | admin123 | 绝密（4级）|
| hr_user | hr123 | 内部（2级）|
| employee | emp123 | 公开（1级）|
        """)


# ─── 主页面 ───────────────────────────────────────────────────────────────────

def render_sidebar():
    user = st.session_state.user
    st.sidebar.markdown(f"### 👤 {user['username']}")
    st.sidebar.markdown(f"- **部门**: {user['department']}")
    st.sidebar.markdown(f"- **用户组**: {user['user_group']}")
    st.sidebar.markdown(f"- **权限**: {permission_badge(user['permission_level'])}")
    st.sidebar.markdown("---")

    # 文档管理（仅高权限用户）
    if user["permission_level"] >= 3:
        st.sidebar.markdown("### 📁 文档管理")

        with st.sidebar.expander("上传文档", expanded=False):
            uploaded_file = st.file_uploader("选择文件（.txt）", type=["txt"])
            doc_title = st.text_input("文档标题")
            perm_level = st.selectbox(
                "权限级别",
                options=[1, 2, 3, 4],
                format_func=lambda x: permission_badge(x),
            )
            user_groups_input = st.text_input("允许用户组（逗号分隔）", "all")
            departments_input = st.text_input("允许部门（逗号分隔）", "all")

            if st.button("上传", use_container_width=True) and uploaded_file and doc_title:
                with st.spinner("向量化中..."):
                    resp = requests.post(
                        f"{API_BASE}/docs/upload",
                        files={"file": (uploaded_file.name, uploaded_file.read(), "text/plain")},
                        data={
                            "title": doc_title,
                            "permission_level": perm_level,
                            "user_groups": user_groups_input,
                            "departments": departments_input,
                        },
                        headers=api_headers(),
                    )
                if resp.status_code == 200:
                    info = resp.json()
                    st.success(f"上传成功！共 {info['chunks']} 个 chunk")
                else:
                    st.error(resp.text)

        st.sidebar.markdown("---")

    # 已有文档列表
    st.sidebar.markdown("### 📚 可访问文档")
    if st.sidebar.button("刷新列表"):
        st.rerun()

    try:
        docs = requests.get(f"{API_BASE}/docs/list", headers=api_headers(), timeout=5).json()
        for doc in docs:
            col_a, col_b = st.sidebar.columns([4, 1])
            col_a.markdown(
                f"**{doc['title']}**  \n{permission_badge(doc['permission_level'])} · {doc['chunks']} chunks"
            )
            if user["user_group"] == "admin":
                if col_b.button("🗑", key=f"del_{doc['doc_id']}"):
                    requests.delete(
                        f"{API_BASE}/docs/{doc['doc_id']}",
                        headers=api_headers(),
                    )
                    st.rerun()
    except Exception:
        st.sidebar.warning("获取文档列表失败")

    st.sidebar.markdown("---")
    if st.sidebar.button("退出登录", use_container_width=True):
        logout()


def render_chat():
    st.markdown("## 💬 知识库对话")
    st.markdown("---")

    # 历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 输入框
    if prompt := st.chat_input("请输入您的问题..."):
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Agent 回答
        with st.chat_message("assistant"):
            full_response = stream_chat(prompt)
            st.markdown(full_response)

        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )

    # 清空对话按钮
    if st.session_state.messages:
        if st.button("🗑 清空对话"):
            st.session_state.messages = []
            st.rerun()


# ─── 路由 ─────────────────────────────────────────────────────────────────────

if st.session_state.token is None:
    render_login()
else:
    render_sidebar()
    render_chat()
