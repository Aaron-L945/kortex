import sys
import os

# 将项目根目录加入路径，确保能导入 app.user_manager
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.user_manager import UserManager
from loguru import logger

def batch_register():
    # 初始化 UserManager (会自动连接 data/users.db)
    user_db = UserManager()
    
    # 定义 10 个测试账户
    # 格式: (用户名, 密码, 部门, 角色)
    test_users = [
        ("aaron_admin",  "Admin@2026", "Tech",    "admin"),
        ("tech_lead",    "TechPass123", "Tech",    "internal"),
        ("tech_dev",     "DevPass456",  "Tech",    "internal"),
        ("sales_mgr",    "SalesMgr99",  "Sales",   "internal"),
        ("sales_rep",    "SalesRep01",  "Sales",   "internal"),
        ("hr_dir",       "HRPass888",   "HR",      "admin"),
        ("hr_staff",     "HRStaff77",   "HR",      "internal"),
        ("fin_head",     "FinMaster1",  "Finance", "admin"),
        ("fin_acc",      "AccPass22",   "Finance", "internal"),
        ("guest_user",   "Guest666",    "General", "internal"),
    ]

    logger.info(f"开始批量注册 {len(test_users)} 个账户...")
    
    success_count = 0
    for username, password, dept, role in test_users:
        # 调用 UserManager 内部的哈希与存库逻辑
        success = user_db.register_user(
            username=username, 
            password=password, 
            dept=dept, 
            role=role
        )
        if success:
            success_count += 1
            logger.info(f"成功注册: {username} | 部门: {dept} | 角色: {role}")
        else:
            logger.warning(f"跳过注册: {username} (可能已存在)")

    logger.success(f"注册流程结束。成功: {success_count}, 失败/跳过: {len(test_users) - success_count}")

if __name__ == "__main__":
    batch_register()