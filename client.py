import streamlit as st
import requests
import json

st.title("企业级安全 RAG 助手")

# 在侧边栏配置 Token
token = st.sidebar.text_input("输入 Access Token", type="password")

if "messages" not in st.session_state:
    st.session_state.messages = []

# 渲染历史对话
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 聊天输入
if prompt := st.chat_input("请输入您的问题..."):
    if not token:
        st.error("请先在侧边栏输入有效的 Token！")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            # 发送请求到你的 Docker 后端
            headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
            data = {"query": prompt, "stream": True}
            
            with requests.post("http://10.66.196.31:8002/v1/chat/completions", 
                               headers=headers, json=data, stream=True) as r:
                for line in r.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith('data: '):
                            content = decoded_line[6:]
                            if content == '[DONE]':
                                break
                            
                            try:
                                result = json.loads(content)
                                # 提取文字块
                                chunk = result['choices'][0]['delta'].get('content', '')
                                full_response += chunk
                                # 实时更新网页内容
                                response_placeholder.markdown(full_response + "▌")
                            except Exception:
                                continue
            
            response_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})