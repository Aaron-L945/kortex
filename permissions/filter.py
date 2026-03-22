from typing import List
from models.schemas import DocumentMetadata, UserInfo


class PermissionFilter:
    """
    权限控制层：对 RAG 召回的 chunks 进行过滤。

    规则：
      1. 文档的 permission_level <= 用户的 permission_level
      2. 文档的 user_groups 包含用户的 user_group，OR 包含 "all"
      3. 文档的 departments 包含用户的 department，OR 包含 "all"
      三条规则同时满足才可访问。
    """

    @staticmethod
    def filter(
        chunks: List[DocumentMetadata],
        user: UserInfo,
    ) -> List[DocumentMetadata]:
        allowed = []
        for chunk in chunks:
            if not PermissionFilter._check(chunk, user):
                continue
            allowed.append(chunk)
        return allowed

    @staticmethod
    def _check(chunk: DocumentMetadata, user: UserInfo) -> bool:
        # 权限等级
        if chunk.permission_level > user.permission_level:
            return False

        # 用户组
        groups_ok = (
            "all" in chunk.user_groups
            or user.user_group in chunk.user_groups
        )
        if not groups_ok:
            return False

        # 部门
        depts_ok = (
            "all" in chunk.departments
            or user.department in chunk.departments
        )
        if not depts_ok:
            return False

        return True
