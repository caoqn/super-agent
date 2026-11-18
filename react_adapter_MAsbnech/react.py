"""ReAct 循环引擎（Thought / Action / Observation）。

遵循 ReAct-master（Yao et al. 2022）的推理-行动交错范式，适配到 chat 模型：
每轮模型输出 "Thought: ...\nAction: ..."，执行 Action 得到 Observation，拼回历史。
工具通过 Action 语法 `tool[arg1,arg2]` 调用；结束通过 `Finish[answer]`。
"""
import re

from llm import complete, Usage


def parse_thought_action(text: str):
    """从模型输出中抽取最后的 Thought 与 Action。

    返回 (thought, action)，任一缺失为 None。
    """
    if not text:
        return None, None
    thoughts = re.findall(r"(?im)^\s*Thought\s*:\s*(.+)$", text)
    actions = re.findall(r"(?im)^\s*Action\s*:\s*(.+)$", text)
    thought = thoughts[-1].strip() if thoughts else None
    action = actions[-1].strip() if actions else None
    return thought, action


def split_action(action: str):
    """解析 Action 字符串为 (tool_name, args_list) 或 ("Finish", [answer])。

    支持 `tool[a,b]`、`tool[]`、`Finish[answer]`（可缺右括号）、`Finish answer` 等。
    """
    if action is None:
        return None, []
    a = action.strip()
    # Finish 变体（兼容缺右括号 / 圆括号 / 冒号 / 无括号）
    if re.match(r"^Finish\b", a, flags=re.I):
        rest = re.sub(r"^Finish\s*", "", a, flags=re.I).strip()
        rest = re.sub(r"^[:：]\s*", "", rest)
        if rest.startswith("[") or rest.startswith("("):
            close = "]" if rest.startswith("[") else ")"
            rest = rest[1:]
            if rest.endswith(close):
                rest = rest[:-1]
        return "Finish", [rest.strip() or a]
    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\[(.*)\]$", a, flags=re.S)
    if m:
        name = m.group(1)
        inner = m.group(2).strip()
        args = [x.strip() for x in inner.split(",")] if inner else []
        return name, args
    # 无括号且非已知工具：整体当作 Finish 的答案
    return "Finish", [a]


def run_react(system_prompt, tools, first_message=None, max_steps=10,
              max_tokens=1024, on_step=None, required_calls=None):
    """执行一次 ReAct 任务。

    参数:
        system_prompt: 系统提示（含任务说明、工具说明、输出格式）
        tools: dict[tool_name -> callable(args) -> observation_str]
        first_message: 首条 user 消息全文（如 "Question: ..." 或 "Observation: ..."），可为 None
        max_steps: 最大 Thought/Action 轮数
        on_step: 可选回调 on_step(step, thought, action, obs)
        required_calls: 可选 [(tool_name, first_arg), ...] 列表，Finish 前必须全部调用过
            （first_arg 用小写归一化）。缺任一项则拦下 Finish，提示补调后继续。
    返回:
        (final_answer, usage, trajectory)
        final_answer 为 Finish[...] 的内容；未 Finish 则为 None
        trajectory 为 list[(thought, action, obs)]
    """
    usage = Usage()
    messages = [{"role": "system", "content": system_prompt}]
    if first_message is not None:
        messages.append({"role": "user", "content": first_message})

    trajectory = []
    final_answer = None
    called = set()  # 已调用的 (tool_name, first_arg_lower)
    for step in range(1, max_steps + 1):
        content, pt, ct = complete(messages, max_tokens=max_tokens)
        usage.add(pt, ct)
        thought, action = parse_thought_action(content)
        if action is None:
            # 模型没给 Action：把整段当 Thought，让其继续
            thought = content[:500]
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": "请按格式输出 Thought 和 Action 两行。"})
            trajectory.append((thought, None, "<no action>"))
            if on_step:
                on_step(step, thought, None, "<no action>")
            continue

        name, args = split_action(action)
        if name == "Finish":
            if required_calls:
                missing = [rc for rc in required_calls if rc not in called]
                if missing:
                    obs = ("不能结束：尚未调用必需工具，仍需调用 "
                           + ", ".join(f"{n}[{a}]" for n, a in missing)
                           + "。请先调用它们拿到证据，再 Finish 提交答案。")
                    trajectory.append((thought, action, obs))
                    if on_step:
                        on_step(step, thought, action, obs)
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content": f"Observation: {obs}"})
                    continue
            final_answer = args[0] if args else ""
            trajectory.append((thought, action, "<finished>"))
            if on_step:
                on_step(step, thought, action, "<finished>")
            break

        first_arg = (args[0] if args else "").strip().lower()
        if name in tools:
            called.add((name, first_arg))
        if name not in tools:
            obs = f"Error: 未知工具 '{name}'。可用工具: {', '.join(tools.keys())}。"
        else:
            try:
                obs = tools[name](args)
                obs = "" if obs is None else str(obs)
            except Exception as e:
                obs = f"Error: 工具 '{name}' 执行异常: {e}"

        trajectory.append((thought, action, obs))
        if on_step:
            on_step(step, thought, action, obs)
        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": f"Observation: {obs}"})

    return final_answer, usage, trajectory
