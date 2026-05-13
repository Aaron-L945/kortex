import streamlit as st
import requests
import json
import httpx
import asyncio
import os
import tempfile
import time
from typing import Dict, List, Any
from loguru import logger

# ==========================================
# 1. 基础配置
# ==========================================
API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="企业级 RAG 助手", layout="wide")

# 初始化 Session State
if "token" not in st.session_state:
    st.session_state.token = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ingest_status" not in st.session_state:
    st.session_state.ingest_status = ""

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
# 3. 文档入库核心逻辑
# ==========================================
def ingest_single_pdf_to_milvus(
    file_bytes: bytes,
    file_name: str,
    access_info: Dict,
    business_tags: Dict,
    embedder: Any,
    progress_callback=None
) -> bool:
    """
    将单个 PDF 文件入库到 Milvus
    """
    import traceback
    from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    def log_only(msg):
        """仅输出到控制台日志，不显示在 UI"""
        logger.info(msg)
    
    def ui_show(msg):
        """输出到 UI 进度显示"""
        if progress_callback:
            progress_callback(msg)
    
    try:
        # ========== 开始入库 ==========
        log_only(f"[入库] 开始处理文件: {file_name}")
        log_only(f"[入库] 文件大小: {len(file_bytes)} bytes")
        
        # 保存临时文件
        log_only(f"[入库] 保存临时文件...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_bytes)
            tmp_path = tmp_file.name
        log_only(f"[入库] 临时文件路径: {tmp_path}")
        
        # 连接 Milvus
        milvus_host = os.getenv("MILVUS_HOST", "127.0.0.1")
        milvus_port = os.getenv("MILVUS_PORT", "19530")
        log_only(f"[入库] 连接 Milvus: {milvus_host}:{milvus_port}")
        connections.connect("default", host=milvus_host, port=milvus_port, timeout=60)
        log_only(f"[入库] Milvus 连接成功")
        
        collection_name = "enterprise_knowledge_vault"
        dim = 1024
        
        # 检查或创建 collection
        from pymilvus import utility
        if utility.has_collection(collection_name):
            log_only(f"[入库] 发现已有 Collection: {collection_name}")
            collection = Collection(collection_name)
            # 获取现有 schema 的字段顺序
            fields_info = collection.schema.fields
            field_names = [f.name for f in fields_info]
            log_only(f"[入库] 现有字段顺序: {field_names}")
        else:
            log_only(f"[入库] 创建新 Collection: {collection_name}")
            fields = [
                FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4000),
                FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="owner_id", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="project_id", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="domain", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="department", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=50, max_length=100),
                FieldSchema(name="role_access", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=50, max_length=100),
            ]
            schema = CollectionSchema(fields, "企业级知识库")
            collection = Collection(collection_name, schema)
            
            index_params = {"metric_type": "COSINE", "index_type": "IVF_FLAT", "params": {"nlist": 1024}}
            collection.create_index("vector", index_params)
            log_only(f"[入库] Collection 创建并建立索引完成")
            fields_info = collection.schema.fields
            field_names = [f.name for f in fields_info]
        
        collection.load()
        log_only(f"[入库] Collection 已加载到内存")
        
        # 加载并切分文档
        log_only(f"[入库] 正在加载 PDF 文档...")
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        log_only(f"[入库] PDF 加载完成，页数: {len(documents)}")
        
        if not documents:
            log_only(f"[入库] 文档为空，跳过")
            os.unlink(tmp_path)
            return False
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
        chunks = splitter.split_documents(documents)
        log_only(f"[入库] 文档切分完成，块数: {len(chunks)}")
        
        texts = [c.page_content.replace('\x00', '').strip()[:3900] for c in chunks if c.page_content.strip()]
        log_only(f"[入库] 有效文本块数: {len(texts)}")
        
        if not texts:
            log_only(f"[入库] 无有效文本，跳过")
            os.unlink(tmp_path)
            return False
        
        # 生成 embeddings
        total_chunks = len(texts)
        log_only(f"[入库] 开始生成 Embeddings，共 {total_chunks} 块...")
        
        all_embeddings = []
        batch_size = 4
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            log_only(f"[入库] embedding 批次 {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            vecs = embedder.embed_documents(batch)
            all_embeddings.extend(vecs)
            time.sleep(0.5)
        
        log_only(f"[入库] Embeddings 生成完成，有效向量数: {len(all_embeddings)}")
        
        # 对齐数据
        valid_texts, valid_vecs = [], []
        for idx, (t, e) in enumerate(zip(texts, all_embeddings)):
            if e is not None:
                vec = e.tolist() if hasattr(e, "tolist") else e
                valid_texts.append(t)
                valid_vecs.append(vec)
            else:
                log_only(f"[入库] 第 {idx} 块向量为空，跳过")
        
        log_only(f"[入库] 有效数据: {len(valid_texts)} 条文本, {len(valid_vecs)} 个向量")
        
        # 准备元数据
        def _to_list(val, default):
            if isinstance(val, list): return [str(i) for i in val]
            return [str(val)] if val else [default]
        
        dept_list = _to_list(access_info.get("department"), "public")
        role_list = _to_list(access_info.get("role_access"), "user")
        log_only(f"[入库] 部门列表: {dept_list}")
        log_only(f"[入库] 角色列表: {role_list}")
        
        # 根据字段名称构建数据字典
        def build_data_for_fields(field_list, current_count):
            """根据字段名称动态构建数据"""
            data = []
            for field in field_list:
                fname = field.name
                ftype = field.dtype
                
                if fname == "vector":
                    data.append(batch_v)
                elif fname == "text":
                    data.append(batch_t)
                elif fname == "file_name":
                    data.append([file_name] * current_count)
                elif fname == "owner_id":
                    data.append([str(access_info.get("owner_id", "admin"))] * current_count)
                elif fname == "project_id":
                    data.append([str(access_info.get("project_id", "default"))] * current_count)
                elif fname == "domain":
                    data.append([str(business_tags.get("domain", "general"))] * current_count)
                elif fname == "doc_type":
                    data.append([str(business_tags.get("doc_type", "PDF"))] * current_count)
                elif fname == "department":
                    data.append([dept_list for _ in range(current_count)])
                elif fname == "role_access":
                    data.append([role_list for _ in range(current_count)])
                elif fname == "pk":
                    # auto_id 字段，跳过
                    pass
                else:
                    log_only(f"[入库] 未知字段 {fname}，跳过")
            
            return data
        
        # 分批插入
        inner_batch_size = 10
        total_inserted = 0
        
        for i in range(0, len(valid_vecs), inner_batch_size):
            batch_v = valid_vecs[i : i + inner_batch_size]
            batch_t = valid_texts[i : i + inner_batch_size]
            current_count = len(batch_v)
            
            log_only(f"[入库] 插入批次 {i//inner_batch_size + 1}")
            
            # 动态构建数据
            data = build_data_for_fields(fields_info, current_count)
            log_only(f"[入库] 准备插入数据，字段数: {len(data)}")
            
            collection.insert(data)
            total_inserted += current_count
            time.sleep(1.5)
        
        log_only(f"[入库] 开始 Flush...")
        collection.flush()
        log_only(f"[入库] Flush 完成")
        
        connections.disconnect("default")
        log_only(f"[入库] Milvus 断开连接")
        
        os.unlink(tmp_path)
        log_only(f"[入库] 临时文件已删除")
        
        # 仅在结束时显示成功信息到 UI
        ui_show(f"✅ {file_name} 入库完成！共 {total_inserted} 条记录")
        return True
        
    except Exception as e:
        error_detail = traceback.format_exc()
        log_only(f"[入库] 入库失败: {str(e)}")
        log_only(f"[入库] 错误详情:\n{error_detail}")
        ui_show(f"❌ 入库失败: {str(e)}")
        return False


# ==========================================
# 4. 聊天界面逻辑
# ==========================================
def chat_page():
    st.title("🤖 企业级安全 RAG 助手")
    
    # 侧边栏：用户信息、文档入库
    with st.sidebar:
        st.write("✅ 已登录")
        
        # ===== 文档入库区域 =====
        with st.expander("📁 文档入库", expanded=False):
            st.markdown("**上传 PDF 文档**")
            
            # 文件上传
            uploaded_file = st.file_uploader(
                "选择 PDF 文件",
                type=["pdf"],
                help="支持单个 PDF 文件上传"
            )
            
            if uploaded_file:
                st.info(f"📄 {uploaded_file.name} ({(uploaded_file.size / 1024):.1f} KB)")
            
            # 配置选项
            st.markdown("**访问权限设置**")
            
            department_options = ["public", "IT_Dept", "HR_Dept", "Finance_Dept", "Sales_Dept"]
            selected_depts = st.multiselect(
                "部门",
                department_options,
                default=["public"],
                help="选择可访问的部门"
            )
            
            role_options = ["user", "admin", "architect", "manager"]
            selected_roles = st.multiselect(
                "角色权限",
                role_options,
                default=["user"],
                help="选择可访问的角色"
            )
            
            # 文档类型
            doc_domain = st.selectbox(
                "文档领域",
                ["general", "tech", "business", "hr", "finance"],
                help="选择文档所属领域"
            )
            
            # 入库按钮
            if st.button("🚀 开始入库", type="primary", disabled=uploaded_file is None):
                if uploaded_file:
                    # 创建进度容器
                    progress_area = st.empty()
                    
                    def update_progress(msg):
                        progress_area.info(msg)
                    
                    update_progress("🔄 正在初始化...")
                    
                    try:
                        from langchain_huggingface import HuggingFaceEmbeddings
                        from dotenv import load_dotenv
                        
                        load_dotenv()
                        embedding_model_path = os.getenv("EMBEDDING_MODEL_NAME")
                        
                        update_progress("📊 正在加载模型...")
                        embedder = HuggingFaceEmbeddings(
                            model_name=embedding_model_path,
                            model_kwargs={"device": "cpu"},
                        )
                        
                        access_info = {
                            "owner_id": "admin",
                            "department": selected_depts,
                            "role_access": selected_roles,
                            "project_id": "default"
                        }
                        
                        business_tags = {
                            "domain": doc_domain,
                            "doc_type": "PDF"
                        }
                        
                        success = ingest_single_pdf_to_milvus(
                            file_bytes=uploaded_file.getvalue(),
                            file_name=uploaded_file.name,
                            access_info=access_info,
                            business_tags=business_tags,
                            embedder=embedder,
                            progress_callback=update_progress
                        )
                        
                        if success:
                            st.success("🎉 文档入库成功！")
                        else:
                            st.error("💔 文档入库失败")
                            
                    except Exception as e:
                        st.error(f"❌ 出错: {str(e)}")
        
        # 分隔线
        st.divider()
        
        # 退出登录
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
            # 构建历史对话格式（排除当前用户消息，因为已在 query 中）
            history = []
            for msg in st.session_state.messages[:-1]:  # 排除最后一条（当前用户消息）
                role = "user" if msg["role"] == "user" else "assistant"
                history.append({"role": role, "content": msg["content"]})
            payload = {"query": prompt, "stream": True, "history": history}
            
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
                                            # 如果是第一个有效字符，清空之前的"正在检索..."提示
                                            if not first_token_received:
                                                full_response = "" 
                                                first_token_received = True
                                            
                                            full_response += chunk
                                            # 实时更新 UI
                                            response_placeholder.markdown(full_response + "▌")
                                            await asyncio.sleep(0.01) 
                                except:
                                    continue
            
            # 运行异步任务（这期间 UI 会一直显示"正在检索..."）
            asyncio.run(get_streaming_response())
            
            # 3. 最后收尾
            if not first_token_received:
                response_placeholder.error("💔 模型响应超时或未返回有效内容。")
            else:
                response_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})

# ==========================================
# 5. 路由控制
# ==========================================
if st.session_state.token is None:
    login_page()
else:
    chat_page()