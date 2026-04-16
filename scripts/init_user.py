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
        # --- 超级管理员 (上帝视角测试) ---
        ("aaron_admin", "Admin@2026", "Tech", "admin"),  # 核心技术管理
        ("hr_dir", "HRPass888", "HR", "admin"),  # HR 部门管理
        # --- 研发部 (测试 RPA 文档的部门匹配) ---
        ("dev_zero", "DevZero001", "Dev", "internal"),  # 属于 Dev 部门，应能看到 RPA 文档
        (
            "dev_coder",
            "DevPass456",
            "Dev",
            "private",
        ),  # 角色为 private，应能精准匹配 RPA 文档
        # --- 财务部 (测试权限阻断) ---
        (
            "fin_head",
            "FinMaster1",
            "Finance",
            "admin",
        ),  # 虽然是 admin，但在当前逻辑下可能搜不到 Tech 档
        ("fin_acc", "AccPass22", "Finance", "internal"),  # 典型的“跨部门”无权限用户
        # --- 访客与公共 (测试 guest 角色) ---
        (
            "visitor_01",
            "Guest666",
            "Public",
            "guest",
        ),  # 专门测试《人物传记》等 guest 文档
        ("outsider", "OutPass99", "General", "guest"),  # 外部人员测试
        # --- 特殊身份 (测试 Owner 匹配) ---
        (
            "user_rpa",
            "RPAPass123",
            "Operations",
            "user",
        ),  # 它是 RPA 文档的 owner，即便部门不匹配也该能看到
    ]

    logger.info(f"开始批量注册 {len(test_users)} 个账户...")

    success_count = 0
    for username, password, dept, role in test_users:
        # 调用 UserManager 内部的哈希与存库逻辑
        success = user_db.register_user(
            username=username, password=password, dept=dept, role=role
        )
        if success:
            success_count += 1
            logger.info(f"成功注册: {username} | 部门: {dept} | 角色: {role}")
        else:
            logger.warning(f"跳过注册: {username} (可能已存在)")

    logger.success(
        f"注册流程结束。成功: {success_count}, 失败/跳过: {len(test_users) - success_count}"
    )


if __name__ == "__main__":
    batch_register()
