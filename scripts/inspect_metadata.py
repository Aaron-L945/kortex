import os
import sys
from pymilvus import connections, Collection

# 确保能找到 core 模块
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

def inspect_all_tags():
    COLLECTION_NAME = "enterprise_knowledge_vault"
    MILVUS_HOST = "10.66.196.31"
    
    connections.connect("default", host=MILVUS_HOST, port="19530")
    collection = Collection(COLLECTION_NAME)
    collection.load()

    # 获取所有不同的文件名
    # 注意：Milvus 不支持直接 distinct，我们先查出所有的文件名
    print(f"\n{'='*100}")
    print(f"{'文件名':<40} | {'Owner':<10} | {'Department':<20} | {'Role Access':<15}")
    print(f"{'-'*100}")

    # 遍历你那 6 个文件，每个文件取 1 条数据看标签
    file_list = [
        "人物传记出版量调研.pdf",
        "RPA+AI重塑职场价值链.pdf",
        "近年畅销书类型.pdf",
        "2025智能体互联网技术白皮书.pdf",
        "Nuage-VSP-20.10.R14.1-AVRS,-VSD,-and-VSC-Solution-Brief-for-Centos-7.6-and-BC-Linux-for-Euler-21.10.pdf",
        "Logging 最佳实践.pdf"
    ]

    for fname in file_list:
        # 使用 query 查询该文件的第一条记录
        res = collection.query(
            expr=f"file_name == '{fname}'",
            output_fields=["file_name", "owner_id", "department", "role_access", "domain", "doc_type"],
            limit=1
        )
        
        if res:
            data = res[0]
            # 格式化输出
            dept_str = str(data.get('department', []))
            role_str = str(data.get('role_access', []))
            owner = data.get('owner_id', 'N/A')
            
            print(f"{fname[:40]:<40} | {owner:<10} | {dept_str:<20} | {role_str:<15}")
        else:
            print(f"{fname[:40]:<40} | ❌ 未能在库中找到该文件的标签信息")

    print(f"{'='*100}\n")

if __name__ == "__main__":
    inspect_all_tags()