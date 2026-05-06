import os
import sys
from pymilvus import connections, Collection
from config import Config

# 确保能找到 core 模块
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

def verify_milvus_data():
    # 1. 配置参数
    COLLECTION_NAME = "enterprise_knowledge_vault"
    DATA_DIR = os.path.join(project_root, "data")
    MILVUS_HOST = Config.MILVUS_HOST
    MILVUS_PORT = "19530"

    print("--- 开始数据一致性检查 ---")

    try:
        # 2. 连接 Milvus
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        
        if not Collection(COLLECTION_NAME):
            print(f"❌ 找不到集合: {COLLECTION_NAME}")
            return

        collection = Collection(COLLECTION_NAME)
        collection.load()

        # 3. 获取本地所有 PDF 文件名
        local_pdfs = [f for f in os.listdir(DATA_DIR) if f.endswith('.pdf')]
        print(f"本地 data/ 目录下共有: {len(local_pdfs)} 个 PDF 文件")

        # 4. 从 Milvus 查询已存在的 file_name 并统计次数
        # 我们查询所有记录，只取 file_name 字段
        results = collection.query(
            expr="pk >= 0", 
            output_fields=["file_name"]
        )

        # 统计数据库中每个文件的分段数
        db_file_counts = {}
        for res in results:
            fname = res['file_name']
            db_file_counts[fname] = db_file_counts.get(fname, 0) + 1

        # 5. 对比结果
        print("\n详细对照表:")
        print(f"{'文件名':<50} | {'数据库分段数':<10} | {'状态'}")
        print("-" * 80)

        missing_count = 0
        for pdf in local_pdfs:
            count = db_file_counts.get(pdf, 0)
            status = "✅ 已同步" if count > 0 else "❌ 未入库"
            if count == 0:
                missing_count += 1
            print(f"{pdf:<50} | {count:<10} | {status}")

        print("-" * 80)
        print(f"检查完毕：{len(local_pdfs) - missing_count} 个已入库，{missing_count} 个缺失。")
        print(f"数据库总实体数 (Entities): {collection.num_entities}")

    except Exception as e:
        print(f"❌ 检查过程中发生错误: {e}")

if __name__ == "__main__":
    verify_milvus_data()