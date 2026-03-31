import re
from collections import Counter

def analyze_bad_cases(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. 提取所有 Bad Case 块
    cases = content.split("-" * 40)
    
    queries = []
    scores = []
    
    for case in cases:
        # 提取 Query
        q_match = re.search(r"【Query】: (.*)", case)
        if q_match:
            queries.append(q_match.group(1))
        
        # 提取分数
        s_match = re.search(r"【Top-1 Score】: ([\d\.]+)", case)
        if s_match:
            scores.append(float(s_match.group(1)))

    if not queries:
        print("未发现有效的坏例数据。")
        return

    # 2. 统计高频词（排除掉常用停用词）
    all_text = " ".join(queries)
    # 简单正则分词，只看长度 > 1 的词
    words = [w for w in re.findall(r"[\u4e00-\u9fa5]{2,}", all_text)]
    word_counts = Counter(words).most_common(10)

    # 3. 计算分数区间
    score_arr = [
        len([s for s in scores if s >= 0.85]),
        len([s for s in scores if 0.75 <= s < 0.85]),
        len([s for s in scores if 0.65 <= s < 0.75])
    ]

    # --- 打印诊断报告 ---
    print("\n" + " 🧠 幻觉原因诊断报告 ".center(50, "="))
    print(f"📊 样本总数: {len(queries)}")
    print(f"📈 分数分布: ")
    print(f"   - 极高分 (>=0.85): {score_arr[0]} 个 (Reranker 严重误判，需加重实体拦截)")
    print(f"   - 中高分 (0.75-0.85): {score_arr[1]} 个")
    print(f"   - 边缘分 (0.65-0.75): {score_arr[2]} 个 (可通过继续提高阈值过滤)")
    
    print(f"\n🔥 幻觉高频词 Top 10:")
    for word, count in word_counts:
        print(f"   - [{word}]: 出现 {count} 次")
    
    print("\n💡 优化建议:")
    if score_arr[0] > len(queries) * 0.3:
        print("   >> 警告：高分幻觉占比大！请检查 pipeline 中的 entities 惩罚是否已生效。")
    if word_counts and word_counts[0][1] > len(queries) * 0.2:
        print(f"   >> 建议：针对词汇 '{word_counts[0][0]}' 建立专门的动作/实体映射表。")
    print("=" * 50)

if __name__ == "__main__":
    # 请替换成你刚才生成的最新文件名
    import os
    files = [f for f in os.listdir('.') if f.startswith('bad_cases_')]
    if files:
        latest_file = sorted(files)[-1]
        analyze_bad_cases(latest_file)
    else:
        print("当前目录下没找到 bad_cases 文件。")