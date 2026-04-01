import json
import sys
from collections import defaultdict

def extract_target_ids_from_file(input_file):
    """
    从输入文件中提取所有target_id
    目标文件格式示例：
    {"query": "...", "target_id": "xxx", "raw_rank": ..., "final_rank": ..., ...}
    """
    target_ids = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                # 解析JSON行
                data = json.loads(line)
                
                # 检查必要的字段
                if 'target_id' in data:
                    target_id = data['target_id']
                    target_ids.append(target_id)
                else:
                    print(f"警告: 第{line_num}行没有target_id字段: {line[:50]}...")
                    
            except json.JSONDecodeError as e:
                print(f"JSON解析错误 (第{line_num}行): {e}")
                print(f"问题行内容: {line[:100]}")
                continue
                
    return target_ids

def search_in_corpus(corpus_file, target_ids):
    """
    在corpus文件中查找所有target_id对应的完整行
    返回字典: {target_id: 完整的json行}
    """
    # 将target_ids转为集合提高查找效率
    target_id_set = set(target_ids)
    results = {}
    found_count = 0
    
    print(f"开始在corpus文件中查找 {len(target_id_set)} 个target_id...")
    
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                
                # 假设corpus文件中的target_id字段名也是'target_id'
                if 'target_id' in data and data['target_id'] in target_id_set:
                    target_id = data['target_id']
                    results[target_id] = {
                        'json_line': line,  # 原始JSON行
                        'data': data,       # 解析后的数据
                        'line_num': line_num # 在corpus中的行号
                    }
                    found_count += 1
                    
                    # 显示进度
                    if found_count % 1000 == 0:
                        print(f"已处理 {line_num:,} 行，找到 {found_count:,} 个匹配项...")
                    
            except json.JSONDecodeError as e:
                # 可以记录错误但继续处理
                if line_num % 100000 == 0:  # 每10万行显示一次错误
                    print(f"corpus文件第{line_num:,}行JSON解析错误: {e}")
                continue
            except Exception as e:
                if line_num % 100000 == 0:
                    print(f"corpus文件第{line_num:,}行处理异常: {e}")
                continue
    
    return results

def main():
    # 文件路径配置
    input_file = "/root/tests/aaron/kortex/failure_analysis.jsonl"  # 你的目标文件，包含target_id
    corpus_file = "/root/tests/aaron/kortex/test_queries.jsonl"  # 要搜索的corpus文件
    output_file = "matched_results.jsonl"  # 输出文件
    
    print("=" * 60)
    print("Target ID 查找工具")
    print("=" * 60)
    
    # 1. 从目标文件提取target_id
    print("\n步骤1: 从目标文件提取target_id...")
    target_ids = extract_target_ids_from_file(input_file)
    
    if not target_ids:
        print("错误: 没有提取到任何target_id，请检查输入文件格式")
        return
    
    print(f"从目标文件提取到 {len(target_ids)} 个target_id")
    
    # 去重
    unique_target_ids = list(set(target_ids))
    print(f"去重后: {len(unique_target_ids)} 个唯一的target_id")
    
    # 如果有重复的target_id
    if len(target_ids) != len(unique_target_ids):
        from collections import Counter
        dup_counts = Counter(target_ids)
        duplicates = {k: v for k, v in dup_counts.items() if v > 1}
        print(f"发现 {len(duplicates)} 个重复的target_id:")
        for tid, count in list(duplicates.items())[:5]:  # 只显示前5个
            print(f"  - {tid}: 出现{count}次")
        if len(duplicates) > 5:
            print(f"  ... 还有 {len(duplicates) - 5} 个")
    
    # 2. 在corpus文件中查找
    print(f"\n步骤2: 在corpus文件中查找...")
    print(f"corpus文件: {corpus_file}")
    
    matched_results = search_in_corpus(corpus_file, unique_target_ids)
    
    # 3. 统计和输出结果
    print("\n" + "=" * 60)
    print("查找结果统计")
    print("=" * 60)
    
    found_ids = set(matched_results.keys())
    not_found_ids = set(unique_target_ids) - found_ids
    
    print(f"输入目标数: {len(unique_target_ids)}")
    print(f"成功匹配: {len(found_ids)}")
    print(f"未找到: {len(not_found_ids)}")
    
    # 4. 保存结果
    if matched_results:
        # 保存原始JSON行
        with open(output_file, 'w', encoding='utf-8') as f:
            for target_id, result_info in matched_results.items():
                f.write(result_info['json_line'] + '\n')
        print(f"\n匹配结果已保存到: {output_file}")
        
        # 保存详细统计信息
        with open("search_statistics.txt", 'w', encoding='utf-8') as f:
            f.write("查找结果统计\n")
            f.write("=" * 40 + "\n")
            f.write(f"输入目标文件: {input_file}\n")
            f.write(f"corpus文件: {corpus_file}\n")
            f.write(f"总目标数: {len(unique_target_ids)}\n")
            f.write(f"成功匹配: {len(found_ids)}\n")
            f.write(f"未找到: {len(not_found_ids)}\n")
            f.write(f"匹配率: {len(found_ids)/len(unique_target_ids)*100:.2f}%\n\n")
            
            if not_found_ids:
                f.write("未找到的target_id列表:\n")
                for tid in sorted(not_found_ids):
                    f.write(f"{tid}\n")
        
        print(f"详细统计已保存到: search_statistics.txt")
        
        # 可选：保存为格式化JSON
        with open("matched_results_formatted.json", 'w', encoding='utf-8') as f:
            formatted_data = []
            for target_id, result_info in matched_results.items():
                formatted_data.append(result_info['data'])
            json.dump(formatted_data, f, ensure_ascii=False, indent=2)
        print(f"格式化结果已保存到: matched_results_formatted.json")
        
    else:
        print("警告: 没有找到任何匹配项!")
    
    if not_found_ids:
        # 保存未找到的ID
        with open("not_found_ids.txt", 'w', encoding='utf-8') as f:
            for tid in sorted(not_found_ids):
                f.write(tid + '\n')
        print(f"未找到的ID列表已保存到: not_found_ids.txt")
        
        # 显示部分未找到的ID
        print(f"\n部分未找到的target_id (前10个):")
        for tid in sorted(list(not_found_ids))[:10]:
            print(f"  - {tid}")

if __name__ == "__main__":
    main()