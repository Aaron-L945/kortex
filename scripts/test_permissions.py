#!/usr/bin/env python3
"""
权限隔离测试脚本

测试 12 个用户账号对文档的访问权限
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pymilvus import connections, Collection
from app.user_manager import UserManager
import sqlite3

MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = 19530

# 测试问题
TEST_QUERY = "RPA 是什么？"

# 文档权限配置（从 Milvus 获取）
DOC_PERMISSIONS = {
    "IPA时代RPA技能.pdf": {
        "owner": "admin",
        "dept": ["IT_Dept", "RPA_Team"],
        "roles": ["user", "developer"]
    },
    "费曼学习法.pdf": {
        "owner": "admin", 
        "dept": ["public"],
        "roles": ["user"]
    }
}

def load_users_from_db():
    """从数据库加载用户"""
    from app.user_manager import UserManager
    manager = UserManager()
    conn = sqlite3.connect(manager.db_path)
    cursor = conn.execute("SELECT username, dept, role FROM users ORDER BY id")
    rows = cursor.fetchall()
    conn.close()
    return rows

# 用户配置
USERS = load_users_from_db()


def get_user_context(username: str, dept: str, role: str) -> dict:
    """构建用户上下文"""
    return {
        "user_id": username,
        "dept": dept,
        "role": role
    }


def build_auth_expr(user_context: dict) -> str:
    """构建权限表达式"""
    if user_context.get("role") == "admin":
        return None  # admin 不过滤
    
    u_id = user_context.get('user_id')
    u_dept = user_context.get('dept')
    u_role = user_context.get('role')
    
    return (
        f"(owner_id == '{u_id}' or "
        f"ARRAY_CONTAINS(department, '{u_dept}') or "
        f"ARRAY_CONTAINS(role_access, '{u_role}'))"
    )


def check_access(user_context: dict) -> list:
    """检查用户能访问的文档"""
    try:
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
        collection = Collection('enterprise_knowledge_vault')
        collection.load()
        
        expr = build_auth_expr(user_context)
        
        # 用一个随机向量测试（只检查权限，不关心结果）
        import numpy as np
        test_vec = np.random.randn(1024).tolist()
        
        results = collection.search(
            data=[test_vec],
            anns_field="vector",
            param={"metric_type": "COSINE", "params": {"nprobe": 16}},
            limit=100,
            expr=expr,
            output_fields=["file_name"]
        )
        
        # 获取唯一文档
        files = set()
        for hit in results[0]:
            files.add(hit.entity.get("file_name"))
        
        return list(files)
    
    finally:
        connections.disconnect("default")


def main():
    print("=" * 70)
    print("🔐 权限隔离测试")
    print("=" * 70)
    
    print("\n📋 文档权限配置：")
    print("-" * 70)
    for doc, perm in DOC_PERMISSIONS.items():
        print(f"  {doc}:")
        print(f"    部门: {perm['dept']}")
        print(f"    角色: {perm['roles']}")
    
    print("\n" + "=" * 70)
    print("👥 用户访问权限测试")
    print("=" * 70)
    
    print(f"\n{'用户名':<15} {'部门':<12} {'角色':<10} {'可访问文档'}")
    print("-" * 70)
    
    for username, dept, role in USERS:
        ctx = get_user_context(username, dept, role)
        accessible_docs = check_access(ctx)
        
        # 格式化输出
        if accessible_docs:
            docs_str = ", ".join(accessible_docs)
        else:
            docs_str = "❌ 无权访问任何文档"
        
        print(f"{username:<15} {dept:<12} {role:<10} {docs_str}")
    
    print("\n" + "=" * 70)
    print("✅ 测试完成")
    print("=" * 70)
    
    # 验证预期
    print("\n📊 权限验证：")
    
    # visitor_01 (public, guest) 应该只能访问费曼学习法
    ctx = get_user_context("visitor_01", "public", "guest")
    docs = check_access(ctx)
    if "费曼学习法.pdf" in docs and "IPA时代RPA技能.pdf" not in docs:
        print("  ✅ visitor_01 (public, guest) - 权限正确")
    else:
        print(f"  ❌ visitor_01 (public, guest) - 权限异常: {docs}")
    
    # admin 应该能访问所有文档
    ctx = get_user_context("admin", "Tech", "admin")
    docs = check_access(ctx)
    if "费曼学习法.pdf" in docs and "IPA时代RPA技能.pdf" in docs:
        print("  ✅ admin - 权限正确")
    else:
        print(f"  ❌ admin - 权限异常: {docs}")
    
    # user_rpa (Operations, user) 应该能访问所有文档
    ctx = get_user_context("user_rpa", "Operations", "user")
    docs = check_access(ctx)
    if "费曼学习法.pdf" in docs and "IPA时代RPA技能.pdf" in docs:
        print("  ✅ user_rpa (Operations, user) - 权限正确")
    else:
        print(f"  ❌ user_rpa (Operations, user) - 权限异常: {docs}")


if __name__ == "__main__":
    main()
