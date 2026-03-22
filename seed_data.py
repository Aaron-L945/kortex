"""
测试数据种子脚本：写入几篇不同权限级别的演示文档。
运行：python seed_data.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from models.schemas import DocumentMetadata
from rag.indexer import FAISSIndexer


SEED_DOCS = [
    # 公开文档（level=1，所有人可访问）
    {
        "title": "公司简介",
        "source": "官网",
        "user_groups": ["all"],
        "departments": ["all"],
        "permission_level": 1,
        "content": """
本公司成立于 2010 年，是一家专注于人工智能和大数据领域的高科技企业。
公司总部位于北京，在上海、深圳设有分公司，员工总数超过 2000 人。
主要产品包括：智能客服系统、知识图谱平台、企业知识库 Agent 等。
公司愿景：用 AI 赋能每一个企业，让知识流动更顺畅。
        """.strip(),
    },
    {
        "title": "员工手册 - 基本规定",
        "source": "HR 文档库",
        "user_groups": ["all"],
        "departments": ["all"],
        "permission_level": 1,
        "content": """
工作时间：周一至周五 9:00-18:00，弹性工作制。
年假政策：入职满 1 年享有 10 天年假，满 3 年 15 天，满 5 年 20 天。
着装要求：商务休闲，重要会议着正装。
行为准则：尊重同事，诚实守信，保护公司资产。
        """.strip(),
    },

    # 内部文档（level=2，需要 permission_level >= 2）
    {
        "title": "2024 年 Q3 销售数据报告",
        "source": "销售部",
        "user_groups": ["admin", "sales", "hr"],
        "departments": ["all"],
        "permission_level": 2,
        "content": """
2024 年第三季度销售总额：1.2 亿元，同比增长 18%。
主要客户：金融行业占比 40%，制造业 30%，零售业 20%，其他 10%。
新签合同数量：85 个，平均合同金额 141 万元。
重点区域：华东市场贡献率最高，达到 45%。
季度目标完成率：102%，超额完成目标。
        """.strip(),
    },
    {
        "title": "技术架构白皮书 v2.0",
        "source": "研发部",
        "user_groups": ["admin", "engineering"],
        "departments": ["engineering", "it"],
        "permission_level": 2,
        "content": """
系统采用微服务架构，核心组件包括：
- API Gateway：基于 Kong，负责流量管理和鉴权
- 知识检索服务：FAISS + BGE 向量模型，支持千万级文档检索
- Agent 服务：LlamaIndex + Claude claude-opus-4-6，支持多轮对话和工具调用
- 向量数据库：FAISS IndexFlatIP，余弦相似度搜索
- 权限服务：基于 RBAC 模型，支持用户组、部门、权限级别三维控制
技术栈：Python 3.11, FastAPI, Streamlit, PostgreSQL, Redis, Docker
        """.strip(),
    },

    # 机密文档（level=3，需要 permission_level >= 3）
    {
        "title": "2025 年战略规划（机密）",
        "source": "战略部",
        "user_groups": ["admin"],
        "departments": ["admin", "strategy"],
        "permission_level": 3,
        "content": """
2025 年战略目标：
1. 营收突破 10 亿元，同比增长 200%
2. 完成 B 轮融资，目标估值 50 亿元
3. 拓展东南亚市场，在新加坡设立亚太区总部
4. 研发投入占比提升至 25%，重点攻关多模态大模型
5. 完成 3-5 起战略并购，补强数据资产和行业场景
核心竞争力建设：构建行业专有大模型，打造数据护城河。
        """.strip(),
    },
]


def seed():
    indexer = FAISSIndexer.get()
    import uuid

    for doc_data in SEED_DOCS:
        doc_id = str(uuid.uuid4())
        # 简单按段落分 chunk
        paragraphs = [p.strip() for p in doc_data["content"].split("\n") if p.strip()]
        chunks = []
        for i, para in enumerate(paragraphs):
            chunks.append(DocumentMetadata(
                doc_id=doc_id,
                chunk_id=f"{doc_id}_{i}",
                title=doc_data["title"],
                source=doc_data["source"],
                content=para,
                user_groups=doc_data["user_groups"],
                departments=doc_data["departments"],
                permission_level=doc_data["permission_level"],
            ))
        indexer.add_chunks(chunks)
        print(f"✅ 已写入：《{doc_data['title']}》 ({len(chunks)} chunks, level={doc_data['permission_level']})")

    print(f"\n🎉 种子数据写入完成，索引总量：{indexer.index.ntotal} 条向量")


if __name__ == "__main__":
    seed()
