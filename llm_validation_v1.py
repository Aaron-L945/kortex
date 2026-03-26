import time
from langchain.prompts import ChatPromptTemplate
from llm_service import get_qwen_llm
from loguru import logger

def run_llm_validation():
    print("\n🤖 " + "=" * 25 + " LLM 逻辑链还原效果验证 " + "=" * 25 + "\n")

    # 1. 模拟刚才验收脚本跑出来的“黄金上下文” (手动精选关键 CID 内容)
    # 这一步是为了排除检索波动，专门测试 LLM 的“智力”
    mock_context = """
--- [文件: Nuage-VSP-20.10.R14.1 | 第 17 页 | CID: 62] ---
10. Remove the drop iptable rule to allow HAProxy distribution to VSD-3: [root@vsd-3 ~]# iptables -D INPUT -s <VSC-IP> -j DROP

--- [文件: Nuage-VSP-20.10.R14.1 | 第 17 页 | CID: 63] ---
Upgrading VSC:
1. Save the existing configurations and take a backup of bof.cfg and config.cfg files from the standby VSC.
• Validate the VSC routes.
2. Copy the 20.10.R14 image to the standby VSC and save the configuration.
3. Reboot the controller.

--- [文件: Nuage-VSP-20.10.R14.1 | 第 17 页 | CID: 64] ---
6. Re-image the hypervisor to BCLinux 21.10 with kernel version 4.19.90... 
7. Copy the XML file and qcow file to the Hypervisor... 
8. virsh define <filename.xml>; virsh start <vm>
9. Verify the VM is spawned...

--- [文件: Nuage-VSP-20.10.R14.1 | 第 17 页 | CID: 65] ---
10. Verify OpenFlow-XMPP communication channel between VSC and AVRS, and VSC and VSD.
11. Verify the BGP neighbor is established.

--- [文件: Nuage-VSP-20.10.R14.1 | 第 17 页 | CID: 66] ---
12. Repeat Step 1 to Step 3 on the primary VSC.
13. Once both the VSCs are upgraded, verify the routes...
    """

    # 2. 构造高强度系统 Prompt
    prompt = ChatPromptTemplate.from_template("""
你是一个严谨的 Nuage 网络技术专家。请根据提供的背景知识，完整整理 VSC 的升级步骤。

【核心要求】：
1. 必须严格按照背景知识中的编号输出。
2. 即使某些步骤（如步骤 11, 12, 13）非常简短，也必须完整列出，严禁省略。
3. 步骤 12 提到的 "Repeat Step 1 to Step 3"，请在回答中明确说明是针对“主用 VSC (Primary VSC)”执行。
4. 必须保留所有 linux 命令（如 iptables）和 virsh 命令。

{style_instruction}

背景知识：
{context}

问题：{question}
""")

    # 3. 初始化 LLM
    try:
        llm = get_qwen_llm(streaming=False)
        chain = prompt | llm
    except Exception as e:
        print(f"❌ LLM 初始化失败: {e}")
        return

    # 4. 执行测试
    query = "如何升级 VSC？请给出完整的 1-13 个步骤。"
    
    print(f"📝 测试 Query: {query}")
    print("⏳ LLM 思考中...\n")

    start_time = time.time()
    response = chain.invoke({
        "context": mock_context,
        "question": query,
        "style_instruction": "请先在 <think> 标签内梳理背景知识中的所有步骤编号，确保 11, 12, 13 没有被遗漏，然后再输出最终答案。"
    })

    # 5. 结果展示
    print("-" * 80)
    print(response.content)
    print("-" * 80)
    
    # 自动化检查
    content = response.content.lower()
    missing = []
    for i in range(1, 14):
        if f"step {i}" not in content and f"{i}." not in content:
            missing.append(str(i))
    
    if not missing:
        print(f"\n🎉 验证成功！LLM 完整识别了 1-13 步。耗时: {time.time()-start_time:.2f}s")
    else:
        print(f"\n❌ 验证失败：缺失步骤 {', '.join(missing)}。需要进一步调整 Prompt 指令。")

if __name__ == "__main__":
    run_llm_validation()
