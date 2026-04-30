import os
from typing import Dict

from pymilvus import connections, Collection, utility
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_NAME")

class RAGExceptionTester:
    def __init__(self):
        connections.connect("default", host="127.0.0.1", port="19530")
        self.collection = Collection("enterprise_knowledge_vault")
        self.embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)

    def run_exception_suite(self):
        print("=== 开始 RAG 系统异常与安全性压力测试 ===\n")

        # --- 场景 1: 越权访问测试 (Horizontal Privilege Escalation) ---
        # 描述：模拟一个外部部门的人员，试图搜索技术部 (Tech) 的 Nuage 手册
        print("🚩 场景 1: 模拟越权访问 (外部人员试图访问技术文档)")
        attacker_context = {"user_id": "bad_actor", "dept": "Sales", "role": "external"}
        res_1 = self._secure_search("Nuage VSP 关键特性", attacker_context)
        
        # 验证：如果结果中出现了 Nuage 相关的文档，则安全性校验失败
        leaked = [hit.entity.get('file_name') for hits in res_1 for hit in hits if "Nuage" in hit.entity.get('file_name')]
        if leaked:
            print(f"❌ 安全漏洞：检测到越权数据泄露! 泄露文件: {leaked}")
        else:
            print("✅ 拦截成功：外部人员无法检索到技术部私有文档。")

        # --- 场景 2: 空查询与极端字符测试 ---
        print("\n🚩 场景 2: 极端输入测试 (空字符串或特殊字符)")
        for edge_query in [" ", "!!! @## @@@", "OR 1=1"]: # 模拟非法输入或类注入字符
            try:
                self._secure_search(edge_query, {"user_id": "admin", "dept": "Tech", "role": "internal"})
                print(f"✅ 稳健性确认：输入 '{edge_query}' 未导致系统崩溃。")
            except Exception as e:
                print(f"⚠️ 系统异常响应: {e}")

        # --- 场景 3: 标签不存在或字段名错误测试 ---
        print("\n🚩 场景 3: 非法业务过滤测试")
        invalid_filter = {"non_exist_field": "some_val"} # 传入一个 Collection 中不存在的列
        try:
            self._secure_search("AVRS", {"user_id": "admin", "dept": "Tech", "role": "internal"}, invalid_filter)
        except Exception as e:
            print(f"✅ 捕获预期错误：非法字段过滤已被 Milvus 拦截。 (Error: {str(e)[:50]}...)")

        # --- 场景 4: 数组越界匹配测试 ---
        print("\n🚩 场景 4: 数组权限边界测试 (空权限数组)")
        # 描述：如果用户的权限标签完全为空，不应返回任何结果
        empty_context = {"user_id": "none", "dept": "None", "role": "None"}
        res_4 = self._secure_search("任何内容", empty_context)
        if len(res_4[0]) == 0:
            print("✅ 权限严密：无权限标签的用户返回 0 条结果。")
        else:
            print(f"⚠️ 逻辑风险：无权限用户检索到了 {len(res_4[0])} 条内容。")

    def _secure_search(self, query: str, user_context: Dict, semantic_filters: Dict = None):
        """复用你的检索逻辑，但增加错误捕捉"""
        query_vec = self.embedder.embed_query(query)
        ac_expr = (
            f"owner_id == '{user_context['user_id']}' or "
            f"ARRAY_CONTAINS(department, '{user_context['dept']}') or "
            f"ARRAY_CONTAINS(role_access, '{user_context['role']}')"
        )
        if semantic_filters:
            for key, val in semantic_filters.items():
                ac_expr = f"({ac_expr}) and ({key} == '{val}')"
        
        return self.collection.search(
            data=[query_vec],
            anns_field="vector",
            param={"metric_type": "L2", "params": {"ef": 64}},
            limit=5,
            expr=ac_expr,
            output_fields=["file_name"]
        )

if __name__ == "__main__":
    tester = RAGExceptionTester()
    tester.run_exception_suite()