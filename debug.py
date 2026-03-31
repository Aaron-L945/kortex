import json
from loguru import logger

def analyze_bad_cases(test_file="test_queries.jsonl", audit_file="retrieval_audit.jsonl"):
    """
    对比测试集和审计日志，找出那 9% 的漏网之鱼
    """
    # 1. 加载测试集映射 (Query -> Target_ID)
    test_map = {}
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            test_map[data["query"]] = {
                "target_id": data["target_id"],
                "target_text": data.get("target_text", "")
            }

    # 2. 读取审计日志
    missed_report = []
    found_queries = set()
    
    with open(audit_file, "r", encoding="utf-8") as f:
        for line in f:
            audit = json.loads(line)
            query = audit["query"]
            if query not in test_map:
                continue
            
            target_id = test_map[query]["target_id"]
            retrieved_ids = [c["docid"] for c in audit["details"]]
            
            # 如果目标 ID 不在前 10 名
            if target_id not in retrieved_ids:
                # 进一步检查：它在不在粗排的原始候选里？（假设你在 audit 里记录了 docid）
                # 这里我们记录关键信息用于复盘
                missed_report.append({
                    "query": query,
                    "expected_id": target_id,
                    "expected_text_snippet": test_map[query]["target_text"][:200],
                    "top_1_retrieved": audit["details"][0] if audit["details"] else "None",
                    "reason": "Not in Top-10" if audit["rough_count"] > 0 else "Zero Recall"
                })

    # 3. 输出分析结果
    with open("debug_missed_cases.json", "w", encoding="utf-8") as f:
        json.dump(missed_report, f, ensure_ascii=False, indent=2)

    print(f"✅ 分析完成！共发现 {len(missed_report)} 个未命中案例。")
    print("💡 请查看 'debug_missed_cases.json' 进行人工对齐。")

if __name__ == "__main__":
    analyze_bad_cases()