import streamlit as st
import requests
import json
import httpx
import asyncio

# ==========================================
# 1. 基础配置
# ==========================================
API_BASE_URL = "http://10.66.196.31:8002"

st.set_page_config(page_title="企业级 RAG 助手", layout="wide")

# 初始化 Session State
if "token" not in st.session_state:
    st.session_state.token = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================================
# 2. 登录逻辑
# ==========================================
def login_page():
    st.title("🔐 企业知识库登录")
    with st.form("login_form"):
        username = st.text_input("用户名")
        password = st.text_input("密码", type="password")
        submitted = st.form_submit_button("登录")
        
        if submitted:
            try:
                # 调用你 FastAPI 的登录接口 (假设路径是 /v1/auth/login)
                resp = requests.post(f"{API_BASE_URL}/v1/auth/login", 
                                     json={"username": username, "password": password})
                if resp.status_code == 200:
                    data = resp.json()
                    st.session_state.token = data.get("access_token")
                    st.success("登录成功！")
                    st.rerun() # 刷新页面进入聊天模式
                else:
                    st.error("用户名或密码错误")
            except Exception as e:
                st.error(f"连接后端失败: {e}")

# ==========================================
# 3. 聊天界面逻辑
# ==========================================
def chat_page():
    st.title("🤖 企业级安全 RAG 助手")
    
    # 侧边栏：用户信息与退出
    with st.sidebar:
        st.write("✅ 已登录")
        if st.button("退出登录"):
            st.session_state.token = None
            st.session_state.messages = []
            st.rerun()

    # 渲染历史对话
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 聊天输入
    if prompt := st.chat_input("请输入您的问题..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # 1. 创建占位符并立即显示提示信息
            response_placeholder = st.empty()
            response_placeholder.markdown("🔍 **正在检索文档并组织语言，请稍候...**")
            
            full_response = ""
            first_token_received = False  # 标记是否收到了第一个有效字符
            
            headers = {
                "Authorization": f"Bearer {st.session_state.token}", 
                "Content-Type": "application/json"
            }
            payload = {"query": prompt, "stream": True}
            
            async def get_streaming_response():
                nonlocal full_response, first_token_received
                async with httpx.AsyncClient(timeout=None) as client:
                    stream_headers = {
                        **headers,
                        "Accept": "text/event-stream",
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                    }
                    
                    async with client.stream("POST", f"{API_BASE_URL}/v1/chat/completions", 
                                           headers=stream_headers, json=payload) as r:
                        if r.status_code == 401:
                            st.error("Token 已失效")
                            return

                        async for line in r.aiter_lines():
                            if line.startswith('data: '):
                                content = line[6:]
                                if content == '[DONE]': break
                                
                                try:
                                    result = json.loads(content)
                                    if "choices" in result:
                                        chunk = result['choices'][0]['delta'].get('content', '')
                                        
                                        # 2. 检查是否有实质性内容输出
                                        if chunk:
                                            # 如果是第一个有效字符，清空之前的“正在检索...”提示
                                            if not first_token_received:
                                                full_response = "" 
                                                first_token_received = True
                                            
                                            full_response += chunk
                                            # 实时更新 UI
                                            response_placeholder.markdown(full_response + "▌")
                                            await asyncio.sleep(0.01) 
                                except:
                                    continue
            
            # 运行异步任务（这期间 UI 会一直显示“正在检索...”）
            asyncio.run(get_streaming_response())
            
            # 3. 最后收尾
            if not first_token_received:
                response_placeholder.error("💔 模型响应超时或未返回有效内容。")
            else:
                response_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})

# ==========================================
# 4. 路由控制
# ==========================================
if st.session_state.token is None:
    login_page()
else:
    chat_page()