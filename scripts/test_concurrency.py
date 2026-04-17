import threading
import requests
import json
import time

# 配置你的后端地址
API_URL = "http://localhost:8000/v1/chat/completions"
# 请确保 Token 有效
TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiYWFyb25fYWRtaW4iLCJkZXB0IjoiVGVjaCIsInJvbGUiOiJhZG1pbiIsImV4cCI6MTc3NjQyNDA4NX0.a7UxA5L4KbwoXKToz94LPM8Lujo97qPfqBNQcefcvD8" 

# 用于存储每个用户的测试结果
results = []
results_lock = threading.Lock()

def fetch_stream(user_id, query):
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "query": query,
        "stream": True
    }
    
    start_time = time.time()
    first_token_time = None
    chunks_count = 0
    status = "Success"
    
    try:
        # 使用 stream=True 保持长连接
        with requests.post(API_URL, json=payload, headers=headers, stream=True, timeout=180) as r:
            if r.status_code != 200:
                print(f"❌ [用户 {user_id}] 失败: HTTP {r.status_code}")
                status = f"HTTP {r.status_code}"
            else:
                for line in r.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith("data: "):
                            content = decoded_line[6:].strip()
                            if content == "[DONE]":
                                break
                            
                            try:
                                data = json.loads(content)
                                # 兼容不同格式的 content 提取
                                text = ""
                                if 'choices' in data:
                                    text = data['choices'][0]['delta'].get('content', '')
                                
                                if text:
                                    chunks_count += 1
                                    if chunks_count == 1:
                                        first_token_time = time.time() - start_time
                                        print(f"⚡ [用户 {user_id}] 首字响应: {first_token_time:.2f}s")
                            except:
                                continue

    except Exception as e:
        print(f"🚨 [用户 {user_id}] 异常: {str(e)}")
        status = f"Error: {type(e).__name__}"

    end_time = time.time()
    total_duration = end_time - start_time
    
    print(f"🏁 [用户 {user_id}] 请求完成 | 总耗时: {total_duration:.2f}s | 块数: {chunks_count}")
    
    # 存入结果汇总
    with results_lock:
        results.append({
            "user": user_id,
            "query": query,
            "ttft": first_token_time, # Time to First Token
            "total": total_duration,
            "chunks": chunks_count,
            "status": status
        })

if __name__ == "__main__":
    tasks = [
        ("User_01", "RPA是什么？"),
        ("User_02", "VSD 如何做升级？"),
        ("User_03", "VSC 如何做升级？"),
        ("User_04", "什么是低代码平台？"),
        ("User_05", "如何配置 Nginx 负载均衡？"),
        ("User_06", "Docker 和虚拟机有什么区别？"),
        ("User_07", "如何优化 Python 并发性能？"),
        ("User_08", "Milvus 向量数据库的原理是什么？"),
        ("User_09", "如何进行数据库分库分表？"),
        ("User_10", "微服务架构的优缺点？"),
    ]
    
    threads = []
    print(f"🔥 开始并发测试 (共 {len(tasks)} 个请求)...")
    global_start = time.time()
    
    for uid, q in tasks:
        t = threading.Thread(target=fetch_stream, args=(uid, q))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    global_duration = time.time() - global_start
    print("\n" + "="*80)
    print(f"📊 并发测试报告汇总 (总执行时长: {global_duration:.2f}s)")
    print(f"{'用户ID':<10} | {'首字耗时(s)':<10} | {'总耗时(s)':<10} | {'数据块数':<10} | {'查询内容'}")
    print("-" * 80)
    
    # 按照耗时排序显示结果
    sorted_results = sorted(results, key=lambda x: x['user'])
    for r in sorted_results:
        ttft_str = f"{r['ttft']:.2f}" if r['ttft'] else "N/A"
        print(f"{r['user']:<10} | {ttft_str:<10} | {r['total']:<10.2f} | {r['chunks']:<10} | {r['query']}")
    
    print("="*80)
    print("✨ 测试结束")