from tools import tools, tool_executor
from core.stream import parse_stream_response, execute_tool_calls
from core.api import call_model_with_retry
from core.memory import ConversationBuffer

def run_agent(user_input: str, memory: ConversationBuffer, system_prompt: str = "你是一个完美的助手。"):
    """处理单次用户输入，返回最终输出（或流式打印）"""
    # 1. 将用户消息加入记忆
    memory.add_user_message(user_input)

    # 2. 构造发送给 API 的消息列表：system + 记忆中的历史

    # 3. 调用模型（可能多轮工具调用）
    while True:
        messages = [{"role": "system", "content": system_prompt}] + memory.get_messages_for_api()
        stream = call_model_with_retry(messages, stream=True, tools=tools)
        content, tool_calls, finish_reason = parse_stream_response(stream, stream_output=True)
        if not tool_calls:
            # 没有工具调用，将助手回复存入记忆并结束
            memory.add_assistant_message({"role": "assistant", "content": content})
            break

        # 有工具调用：执行工具，并将 assistant 消息和 tool 结果加入记忆
        new_messages = execute_tool_calls(content, tool_calls, tool_executor)
        # new_messages[0] 是 assistant 消息（含 tool_calls）
        # new_messages[1:] 是 tool 消息
        memory.add_assistant_message(new_messages[0])
        for tool_msg in new_messages[1:]:
            memory.add_tool_message(tool_msg["tool_call_id"], tool_msg["content"])

    # 可选：返回最终回复文本
    return content

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
        # run_agent 内部已流式打印，这里只需换行（流式输出已经打印完）
        print()  # 确保换行

if __name__ == "__main__":
    main()
