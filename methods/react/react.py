"""ReAct loop engine (Thought / Action / Observation).

Follows the interleaved reasoning/action pattern from ReAct (Yao et al., 2022), adapted to chat models.
Each turn emits "Thought: ...\nAction: ..."; the action is executed and its observation is appended to history.
Tools use the `tool[arg1,arg2]` action syntax; a task ends with `Finish[answer]`.
"""
import re

from .llm import complete, Usage


def parse_thought_action(text: str):
    """Extract the last Thought and Action from model output.

    Returns (thought, action); either value may be None.
    """
    if not text:
        return None, None
    thoughts = re.findall(r"(?im)^\s*Thought\s*:\s*(.+)$", text)
    actions = re.findall(r"(?im)^\s*Action\s*:\s*(.+)$", text)
    thought = thoughts[-1].strip() if thoughts else None
    action = actions[-1].strip() if actions else None
    return thought, action


def split_action(action: str):
    """Parse an action string into (tool_name, args_list) or ("Finish", [answer]).

    Supports `tool[a,b]`, `tool[]`, `Finish[answer]` (with an optional closing bracket), and `Finish answer`.
    """
    if action is None:
        return None, []
    a = action.strip()
    # Accept Finish forms with brackets, parentheses, a colon, or no delimiter.
    if re.match(r"^Finish\b", a, flags=re.I):
        rest = re.sub(r"^Finish\s*", "", a, flags=re.I).strip()
        rest = re.sub(r"^[:\uFF1A]\s*", "", rest)
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
    # Treat an unbracketed action that is not a known tool as the final answer.
    return "Finish", [a]


def run_react(system_prompt, tools, first_message=None, max_steps=10,
              max_tokens=1024, on_step=None, required_calls=None):
    """Run one ReAct task.

    Args:
        system_prompt: System prompt with task instructions, tool descriptions, and output format.
        tools: dict[tool_name -> callable(args) -> observation_str]
        first_message: Optional full first user message, such as "Question: ..." or "Observation: ...".
        max_steps: Maximum number of Thought/Action turns.
        on_step: Optional callback on_step(step, thought, action, obs).
        required_calls: Optional [(tool_name, first_arg), ...] calls required before Finish.
            First arguments are lowercased; missing calls block Finish and prompt the model to continue.
    Returns:
        (final_answer, usage, trajectory)
        final_answer is the content of Finish[...], or None if Finish was not emitted.
        trajectory is a list of (thought, action, observation) tuples.
    """
    usage = Usage()
    messages = [{"role": "system", "content": system_prompt}]
    if first_message is not None:
        messages.append({"role": "user", "content": first_message})

    trajectory = []
    final_answer = None
    called = set()  # Called (tool_name, lowercased first argument) pairs.
    for step in range(1, max_steps + 1):
        content, pt, ct = complete(messages, max_tokens=max_tokens)
        usage.add(pt, ct)
        thought, action = parse_thought_action(content)
        if action is None:
            # If no Action was emitted, treat the response as a Thought and ask the model to continue.
            thought = content[:500]
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": "Please respond with two lines in this format: Thought: ... and Action: ..."})
            trajectory.append((thought, None, "<no action>"))
            if on_step:
                on_step(step, thought, None, "<no action>")
            continue

        name, args = split_action(action)
        if name == "Finish":
            if required_calls:
                missing = [rc for rc in required_calls if rc not in called]
                if missing:
                    obs = ("Cannot finish yet: required tools still need to be called: "
                           + ", ".join(f"{n}[{a}]" for n, a in missing)
                           + ". Call them to gather evidence, then submit your answer with Finish.")
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
            obs = f"Error: unknown tool '{name}'. Available tools: {', '.join(tools.keys())}."
        else:
            try:
                obs = tools[name](args)
                obs = "" if obs is None else str(obs)
            except Exception as e:
                obs = f"Error: tool '{name}' raised an exception: {e}"

        trajectory.append((thought, action, obs))
        if on_step:
            on_step(step, thought, action, obs)
        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": f"Observation: {obs}"})

    return final_answer, usage, trajectory
