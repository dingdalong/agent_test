from tools import tools, tool_executor
from core.api import call_model_with_retry, execute_tool_calls
from memory.memory import ConversationBuffer, VectorMemory
from config import USER_ID

# 初始化长期记忆（每个用户一个集合，这里简单起见使用固定集合）
vector_memory = VectorMemory(collection_name=USER_ID)  # 实际中应根据 user_id 动态选择

def run_agent(user_input: str, memory: ConversationBuffer, system_prompt: str = "你是一个完美的助手。"):
    """处理单次用户输入，集成长期记忆检索和存储"""
    # --- 检索相关记忆 ---
    relevant_memories = vector_memory.search(user_input, n_results=3)
    if relevant_memories:
        # 将记忆拼接到 system prompt 中，或作为额外上下文插入
        memory_context = "以下是你知道的关于用户的一些信息：\n" + "\n".join([m["fact"] for m in relevant_memories])
        # 我们可以把记忆作为 system 消息的一部分，也可以作为单独的 user 消息（但注意 role）
        # 这里我们扩展 system_prompt
        enhanced_system = system_prompt + "\n\n" + memory_context
    else:
        enhanced_system = system_prompt

    # 1. 将用户消息加入短期记忆
    memory.add_user_message(user_input)

    # 2. 多轮工具调用循环
    while True:
        messages = [{"role": "system", "content": enhanced_system}] + memory.get_messages_for_api()
        content, tool_calls, finish_reason = call_model_with_retry(messages, stream=True, tools=tools)

        if not tool_calls:
            # 没有工具调用，将助手回复存入短期记忆并结束
            memory.add_assistant_message({"role": "assistant", "content": content})
            final_response = content
            break

        # 有工具调用：执行工具，并将 assistant 消息和 tool 结果加入短期记忆
        new_messages = execute_tool_calls(content, tool_calls, tool_executor)
        memory.add_assistant_message(new_messages[0])
        for tool_msg in new_messages[1:]:
            memory.add_tool_message(tool_msg["tool_call_id"], tool_msg["content"])

    vector_memory.add_conversation(user_input, final_response)

    return final_response

def main():
    memory = ConversationBuffer(max_rounds=10)
    system_prompt = "你是一个完美的助手。"

    print("Agent 已启动，输入 'exit' 退出。")
    while True:
        user_input = input("\n你: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        print("助手: ", end="", flush=True)
        run_agent(user_input, memory, system_prompt)
        print()  # 确保换行

if __name__ == "__main__":
    main()
