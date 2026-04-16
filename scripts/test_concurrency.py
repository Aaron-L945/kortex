import threading
import requests
import json
import time

# 配置你的后端地址
API_URL = "http://localhost:8000/v1/chat/completions"
# 请替换为你登录后获取的有效 Token
TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiYWFyb25fYWRtaW4iLCJkZXB0IjoiVGVjaCIsInJvbGUiOiJhZG1pbiIsImV4cCI6MTc3NjMzMzQ4MX0.JYOJFXBrSD0SMsOTub1C7LPzoW5-pZ-SsWAhAiyPJMI" 

def fetch_stream(user_id, query):
    print(f"🚀 [用户 {user_id}] 发起请求: {query}")
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "query": query,
        "stream": True
    }
    
    start_time = time.time()
    chunks_count = 0
    
    try:
        # 使用 stream=True 保持长连接
        with requests.post(API_URL, json=payload, headers=headers, stream=True, timeout=180) as r:
            if r.status_code != 200:
                print(f"❌ [用户 {user_id}] 失败: HTTP {r.status_code} - {r.text}")
                return

            for line in r.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith("data: "):
                        content = decoded_line[6:]
                        if content == "[DONE]":
                            break
                        
                        # 解析 JSON
                        try:
                            data = json.loads(content)
                            text = data['choices'][0]['delta'].get('content', '')
                            if text:
                                chunks_count += 1
                                if chunks_count == 1:
                                    print(f"✅ [用户 {user_id}] 收到首字! 耗时: {time.time() - start_time:.2f}s")
                        except:
                            pass

        print(f"🏁 [用户 {user_id}] 请求完成，共收到 {chunks_count} 个数据块")

    except Exception as e:
        print(f"🚨 [用户 {user_id}] 发生异常: {str(e)}")

if __name__ == "__main__":
    # 准备两个不同的问题
    tasks = [
        ("User_01", "RPA是什么？"),
        # ("User_02", "VSD 如何做升级？"),
        # ("User_03", "VSC 如何做升级？"),
        # ("User_04", "什么是低代码平台？"),
        # ("User_05", "如何配置 Nginx 负载均衡？"),
        # ("User_06", "Docker 和虚拟机有什么区别？"),
        # ("User_07", "如何优化 Python 并发性能？"),
        # ("User_08", "Milvus 向量数据库的原理是什么？"),
        # ("User_09", "如何进行数据库分库分表？"),
        # ("User_10", "微服务架构的优缺点？"),
    ]
    threads = []
    print("🔥 开始并发测试...")
    
    for uid, q in tasks:
        t = threading.Thread(target=fetch_stream, args=(uid, q))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print("✨ 测试结束")