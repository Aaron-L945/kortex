#!/usr/bin/env python3
"""
Milvus 文档入库检查脚本

检查 Milvus 中入库的文档及其权限配置
"""

import sys
import os
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
from dotenv import load_dotenv
load_dotenv()

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))


def get_all_files():
    """获取 Milvus 中所有文档及其权限"""
    try:
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    except Exception as e:
        print(f"❌ 连接 Milvus 失败: {e}")
        return
    
    try:
        # 获取所有 collection
        from pymilvus import utility
        collections = utility.list_collections()
        
        if not collections:
            print("📭 Milvus 中没有 collection")
            return
        
        print(f"\n📚 Milvus 中的 Collection ({len(collections)} 个):")
        print("=" * 70)
        
        for coll_name in collections:
            collection = Collection(coll_name)
            collection.load()
            
            # 获取 schema
            schema = collection.schema
            print(f"\n📦 Collection: {coll_name}")
            print(f"   片段数: {collection.num_entities}")
            
            # 查询唯一文件
            results = collection.query(
                expr="pk >= 0",
                output_fields=["file_name", "owner_id", "department", "role_access"],
                limit=1000
            )
            
            # 提取唯一文件
            files = {}
            for r in results:
                fname = r.get("file_name", "Unknown")
                if fname not in files:
                    files[fname] = {
                        "owner": r.get("owner_id", "N/A"),
                        "dept": r.get("department", []),
                        "roles": r.get("role_access", []),
                    }
            
            if files:
                print(f"\n   📄 文档列表 ({len(files)} 个):")
                print(f"   {'文件名称':<40} {'所有者':<12} {'部门':<15} {'角色权限'}")
                print("   " + "-" * 90)
                for fname, info in sorted(files.items()):
                    dept_str = ",".join(info["dept"]) if info["dept"] else "N/A"
                    roles_str = ",".join(info["roles"]) if info["roles"] else "N/A"
                    print(f"   {fname:<40} {info['owner']:<12} {dept_str:<15} {roles_str}")
            
            # 统计权限分布
            all_depts = set()
            all_roles = set()
            for info in files.values():
                all_depts.update(info["dept"] or [])
                all_roles.update(info["roles"] or [])
            
            if all_depts:
                print(f"\n   📊 部门权限: {', '.join(sorted(all_depts))}")
            if all_roles:
                print(f"   📊 角色权限: {', '.join(sorted(all_roles))}")
    
    except Exception as e:
        print(f"❌ 查询失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        connections.disconnect("default")


def main():
    print("🔍 Milvus 文档入库检查")
    print(f"   Milvus: {MILVUS_HOST}:{MILVUS_PORT}")
    print()
    get_all_files()
    print()


if __name__ == "__main__":
    main()
