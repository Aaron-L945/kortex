import json
import asyncio
import random
import os
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI

# 配置
API_KEY = "YOUR_API_KEY"
BASE_URL = "http://10.66.196.31:20201/v1"
MODEL_NAME = "/models/gpt-oss-20b"
CONCURRENT_REQUESTS = 10 

client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

async def call_model(semaphore, item):
    async with semaphore:
        text = item['text']
        title = item.get('title', '')
        prompt = f"你是一个搜索模拟器。根据内容生成一个简短、自然的中文搜索提问，必须包含主语：\n正文：{text[:500]}"
        
        for _ in range(3):
            try:
                response = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    timeout=30.0
                )
                query = response.choices[0].message.content.strip().replace('"', '')
                return {
                    "query": query,
                    "target_id": item['docid'],
                    "target_text": text
                }
            except Exception:
                await asyncio.sleep(1)
        return None

async def main(input_jsonl, output_jsonl, total_target_size=500):
    # 1. 加载已有结果，实现“断点续传”逻辑
    existing_ids = set()
    if os.path.exists(output_jsonl):
        with open(output_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    existing_ids.add(json.loads(line)['target_id'])
                except: continue
        print(f"📁 检测到已有进度，已生成: {len(existing_ids)} 条")

    # 2. 加载原始语料
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        all_data = [json.loads(line) for line in f]

    # 3. 过滤掉已经生成的 docid
    remaining_data = [item for item in all_data if item['docid'] not in existing_ids]
    
    # 计算还需要生成多少条
    needed_count = total_target_size - len(existing_ids)
    if needed_count <= 0:
        print("✅ 已达到目标数量，无需生成。")
        return

    # 从剩余数据中随机抽取
    test_samples = random.sample(remaining_data, min(len(remaining_data), needed_count))
    print(f"🚀 本次任务计划新生成: {len(test_samples)} 条")

    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    tasks = [call_model(semaphore, item) for item in test_samples]
    
    # 4. 以追加模式 (a) 写入文件，保证实时保存
    with open(output_jsonl, 'a', encoding='utf-8') as f_out:
        for f_task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="⚡ 正在累计生成"):
            res = await f_task
            if res:
                f_out.write(json.dumps(res, ensure_ascii=False) + '\n')
                f_out.flush() # 强制刷入磁盘，防止崩溃丢失

    print(f"\n✅ 任务完成！当前文件总数: {len(existing_ids) + len(test_samples)} 条")

if __name__ == "__main__":
    # 参数说明：输入语料, 输出文件, 最终想要的 query 总量
    asyncio.run(main("corpus_dedup.jsonl", "test_queries.jsonl", total_target_size=500))