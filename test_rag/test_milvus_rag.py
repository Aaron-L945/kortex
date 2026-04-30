import os
from pymilvus import connections, Collection, utility
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from config import Config

# 加载配置
load_dotenv()
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_NAME")

def run_test():
    # 1. 连接配置
    host = Config.MILVUS_HOST
    port = "19530"
    collection_name = "enterprise_knowledge_vault"

    print(f"--- 正在连接 Milvus ({host}:{port}) ---")
    try:
        connections.connect("default", host=host, port=port)
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return

    # 2. 检查集合是否存在
    if not utility.has_collection(collection_name):
        print(f"❌ 集合 '{collection_name}' 不存在，请先运行入库脚本。")
        return

    collection = Collection(collection_name)
    collection.flush() # 确保数据已落盘
    print(f"✅ 集合状态: 已就绪")
    print(f"📊 当前总 Entity 数量: {collection.num_entities}")

    # 3. 加载模型 (用于将测试问题向量化)
    print(f"--- 正在加载 Embedding 模型 ---")
    embedder = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_PATH,
        model_kwargs={'device': 'cpu'} # 如果没显卡改为 'cpu'
    )

    # 4. 模拟不同身份的用户进行检索测试
    test_cases = [
        {
            "name": "管理员测试 (admin)",
            "context": {"user_id": "admin", "dept": "Tech", "role": "internal"},
            "query": "费曼学习法的核心步骤是什么？"
        },
        {
            "name": "普通用户测试 (游客)",
            "context": {"user_id": "guest_user", "dept": "Public", "role": "guest"},
            "query": "费曼学习法的核心步骤是什么？"
        }
    ]

    for case in test_cases:
        print(f"\n🚀 运行测试项: {case['name']}")
        print(f"🔍 查询语句: {case['query']}")
        
        # 向量化问题
        query_vec = embedder.embed_query(case['query'])

        # 构造权限表达式 (必须与入库时的逻辑一致)
        user_ctx = case['context']
        ac_expr = (
            f"owner_id == '{user_ctx['user_id']}' or "
            f"ARRAY_CONTAINS(department, '{user_ctx['dept']}') or "
            f"ARRAY_CONTAINS(role_access, '{user_ctx['role']}')"
        )

        # 执行检索
        results = collection.search(
            data=[query_vec],
            anns_field="vector",
            param={"metric_type": "COSINE", "params": {"nprobe": 16}},
            limit=3,
            expr=ac_expr,
            output_fields=["text", "file_name", "department"]
        )

        # 打印结果
        if not results or len(results[0]) == 0:
            print("⚠️ 未匹配到任何结果（可能权限不符或无相关内容）")
        else:
            for i, hit in enumerate(results[0]):
                print(f"  [{i+1}] 相似度: {hit.distance:.4f}")
                print(f"      来源文件: {hit.entity.get('file_name')}")
                print(f"      所属部门: {hit.entity.get('department')}")
                print(f"      内容预览: {hit.entity.get('text')[:100]}...")

if __name__ == "__main__":
    run_test()