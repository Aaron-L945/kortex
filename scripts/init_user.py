#!/usr/bin/env python3
"""
用户初始化脚本 - 向 SQLite 数据库添加用户

使用方法:
    python scripts/init_user.py                    # 交互式添加
    python scripts/init_user.py admin admin123 Tech admin  # 命令行添加
"""

import sys
import os
import argparse

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.user_manager import UserManager


def add_user(username: str, password: str, dept: str, role: str = "internal"):
    """添加单个用户"""
    manager = UserManager()
    
    # 检查用户是否已存在
    import sqlite3
    conn = sqlite3.connect(manager.db_path)
    cursor = conn.execute("SELECT id FROM users WHERE username = ?", (username,))
    if cursor.fetchone():
        print(f"❌ 用户 {username} 已存在！")
        conn.close()
        return False
    
    # 添加用户
    success = manager.register_user(username, password, dept, role)
    conn.close()
    
    if success:
        print(f"✅ 用户 {username} 添加成功！")
        print(f"   部门: {dept}")
        print(f"   角色: {role}")
        return True
    else:
        print(f"❌ 用户 {username} 添加失败！")
        return False


def interactive_mode():
    """交互式添加用户"""
    print("\n📝 添加新用户")
    print("=" * 40)
    
    username = input("用户名: ").strip()
    if not username:
        print("❌ 用户名不能为空")
        return
    
    password = input("密码: ").strip()
    if not password:
        print("❌ 密码不能为空")
        return
    
    print("\n部门选项:")
    print("  1. Tech (技术部)")
    print("  2. HR (人力资源)")
    print("  3. Finance (财务部)")
    print("  4. Sales (销售部)")
    print("  5. public (公共)")
    print("  6. 自定义")
    
    dept_choice = input("选择部门 [1-6]: ").strip()
    dept_map = {"1": "Tech", "2": "HR", "3": "Finance", "4": "Sales", "5": "public"}
    dept = dept_map.get(dept_choice, "Tech")
    if dept_choice == "6":
        dept = input("输入部门名称: ").strip() or "Tech"
    
    print("\n角色选项:")
    print("  1. admin (管理员)")
    print("  2. internal (内部用户)")
    print("  3. user (普通用户)")
    
    role_choice = input("选择角色 [1-3]: ").strip()
    role_map = {"1": "admin", "2": "internal", "3": "user"}
    role = role_map.get(role_choice, "internal")
    
    print()
    add_user(username, password, dept, role)


def list_users():
    """列出所有用户"""
    manager = UserManager()
    import sqlite3
    
    conn = sqlite3.connect(manager.db_path)
    cursor = conn.execute("SELECT id, username, dept, role FROM users")
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        print("📭 暂无用户")
        return
    
    print("\n👥 用户列表")
    print("=" * 50)
    print(f"{'ID':<4} {'用户名':<15} {'部门':<10} {'角色':<10}")
    print("-" * 50)
    for row in rows:
        print(f"{row[0]:<4} {row[1]:<15} {row[2]:<10} {row[3]:<10}")


def delete_user(username: str):
    """删除用户"""
    manager = UserManager()
    import sqlite3
    
    conn = sqlite3.connect(manager.db_path)
    cursor = conn.execute("SELECT id FROM users WHERE username = ?", (username,))
    if not cursor.fetchone():
        print(f"❌ 用户 {username} 不存在！")
        conn.close()
        return False
    
    conn.execute("DELETE FROM users WHERE username = ?", (username,))
    conn.commit()
    conn.close()
    print(f"✅ 用户 {username} 已删除")
    return True


def main():
    parser = argparse.ArgumentParser(description="用户管理脚本")
    parser.add_argument("action", nargs="?", choices=["add", "list", "delete"], 
                        help="操作: add(添加) / list(列表) / delete(删除)")
    parser.add_argument("username", nargs="?", help="用户名")
    parser.add_argument("password", nargs="?", help="密码")
    parser.add_argument("dept", nargs="?", help="部门")
    parser.add_argument("role", nargs="?", help="角色 (默认: internal)")
    
    args = parser.parse_args()
    
    if not args.action or args.action == "list":
        list_users()
    elif args.action == "add":
        if not args.username:
            interactive_mode()
        else:
            if not args.password or not args.dept:
                print("❌ 添加用户需要: username password dept")
                print("   示例: python scripts/init_user.py add admin admin123 Tech admin")
                sys.exit(1)
            role = args.role or "internal"
            add_user(args.username, args.password, args.dept, role)
    elif args.action == "delete":
        if not args.username:
            username = input("输入要删除的用户名: ").strip()
            if username:
                delete_user(username)
        else:
            delete_user(args.username)


if __name__ == "__main__":
    main()
