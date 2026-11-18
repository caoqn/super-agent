import os
import re
import sys
import json
import glob
import math
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from openai import OpenAI  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    OpenAI = None  # type: ignore

import requests

try:
    import httpx  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    httpx = None  # type: ignore

client = None
api_key = ""
base_url = ""
model_name = ""
thinking_override: bool | None = None
max_output_tokens: int | None = None


def _request_options(enable_thinking: bool, *, json_output: bool = False) -> dict:
    """Preserve experiment defaults while allowing bounded smoke-test calls."""
    thinking = enable_thinking if thinking_override is None else thinking_override
    options = {}
    if model_name.startswith(("qwen", "deepseek")):
        if thinking or thinking_override is not None:
            options["extra_body"] = {"enable_thinking": thinking}
    elif thinking and model_name.startswith("gpt"):
        options["extra_body"] = {"reasoning_effort": "high"}
    if json_output and not (
        thinking and model_name.startswith(("qwen", "deepseek", "gpt"))
    ):
        options["response_format"] = {"type": "json_object"}
    if max_output_tokens is not None:
        options["max_tokens"] = max_output_tokens
    return options


def _load_dotenv_file(path: str) -> dict[str, str]:
    """
    Minimal .env parser (KEY=VALUE per line).
    - ignores blank lines and comments (# ...)
    - strips surrounding quotes
    """
    out: dict[str, str] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                key = k.strip()
                val = v.strip().strip('"').strip("'")
                if key:
                    out[key] = val
    except FileNotFoundError:
        return {}
    return out


def _resolve_provider_prefix(provider: str) -> str:
    p = (provider or "").strip().lower()
    if p in ("apiyi", "api_yi", "api-yi"):
        return "API_YI"
    if p in ("sunapi", "sun_api", "sun-api"):
        return "SUNAPI"
    if p in ("qwen", "dashscope", "aliyun"):
        return "QWEN"
    return (provider or "").strip().upper()


def _get_provider_config(provider: str, dotenv_vars: dict[str, str]) -> tuple[str, str]:
    prefix = _resolve_provider_prefix(provider)
    key_name = f"{prefix}_OPENAI_API_KEY"
    base_name = f"{prefix}_OPENAI_BASE"
    k = os.getenv(key_name) or dotenv_vars.get(key_name) or ""
    b = os.getenv(base_name) or dotenv_vars.get(base_name) or ""
    if not k or not b:
        raise ValueError(
            f"Missing provider config for '{provider}'. "
            f"Expected {key_name} and {base_name} in .env (or environment variables)."
        )
    return k, b

token_cost = 0.0
TOKEN_STATS = {
    "total_prompt": 0,
    "total_completion": 0,
    "total_completion_reasoning": 0,
    "total_completion_visible": 0,
    "total_total": 0,
    "by_phase": {},
}



def _get_usage_fields(response) -> tuple[int, int, int]:
    usage = None
    if isinstance(response, dict):
        usage = response.get("usage")
    else:
        usage = getattr(response, "usage", None)

    prompt_tokens = completion_tokens = reasoning_tokens = 0
    if not usage:
        return 0, 0, 0

    if isinstance(usage, dict):
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        details = usage.get("completion_tokens_details") or {}
        if isinstance(details, dict):
            reasoning_tokens = int(details.get("reasoning_tokens") or 0)
        return prompt_tokens, completion_tokens, reasoning_tokens

    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    details = getattr(usage, "completion_tokens_details", None)
    if details is not None:
        reasoning_tokens = int(getattr(details, "reasoning_tokens", 0) or 0)
    return prompt_tokens, completion_tokens, reasoning_tokens


def _get_message_content(response) -> str:
    if isinstance(response, dict):
        try:
            text = response.choices[0].message.content
            if hasattr(response.choices[0].message, 'model_extra') and 'reasoning_content' in response.choices[0].message.model_extra:
                reasoning_content = response.choices[0].message.model_extra['reasoning_content']
                text = text.removeprefix(reasoning_content).lstrip() if reasoning_content else text
            if '<think>' in text and '</think>' in text:
                text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
            return text
        except Exception:
            return ""

    try:
        text = response.choices[0].message.content
        if hasattr(response.choices[0].message, 'model_extra') and 'reasoning_content' in response.choices[0].message.model_extra:
            reasoning_content = response.choices[0].message.model_extra['reasoning_content']
            text = text.removeprefix(reasoning_content).lstrip() if reasoning_content else text
        if '<think>' in text and '</think>' in text:
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        return text
    except Exception:
        return ""


def ensure_logs_dir(base_dir: str) -> str:
    logs_dir = os.path.join(base_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    return os.path.abspath(logs_dir)


def call_llm(prompt: str, enable_thinking: bool = False, **meta) -> str:
    for _ in range(3):
        try:
            kwargs = _request_options(enable_thinking)
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
            global token_cost, TOKEN_STATS
            pt, ct, cr = _get_usage_fields(response)
            cv = max(ct - cr, 0)
            tt = pt + ct
            token_cost += tt

            TOKEN_STATS["total_prompt"] += pt
            TOKEN_STATS["total_completion"] += ct
            TOKEN_STATS["total_completion_reasoning"] += cr
            TOKEN_STATS["total_completion_visible"] += cv
            TOKEN_STATS["total_total"] += tt

            phase = meta.get("phase") or "unknown"
            phase_stats = TOKEN_STATS["by_phase"].setdefault(
                phase,
                {"prompt": 0, "completion": 0, "completion_reasoning": 0, "completion_visible": 0, "total": 0},
            )
            phase_stats["prompt"] += pt
            phase_stats["completion"] += ct
            phase_stats["completion_reasoning"] += cr
            phase_stats["completion_visible"] += cv
            phase_stats["total"] += tt

            return _get_message_content(response).strip()
        except Exception as e:
            print(f"Error calling LLM: {e}")
    return ""

def call_llm_stream(prompt: str, temperature: float = -1, enable_thinking: bool = False, **meta) -> dict:
    """Call LLM and parse a JSON object from response."""
    for _ in range(5):
        try:
            kwargs = _request_options(enable_thinking)
            if temperature != -1:
                kwargs["temperature"] = temperature

            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                stream_options={"include_usage": True},
                **kwargs
            )
            final_reasoning = ""
            result = ""
            usage_info = None
            try:

                for chunk in response:

                    if not chunk.choices:
                        continue

                    if hasattr(chunk, 'usage') and chunk.usage:
                        usage_info = chunk.usage
                        print(f"\n✅ Token usage received: {usage_info}")
                        continue

                    delta = chunk.choices[0].delta



                    reasoning = getattr(delta, 'reasoning_content', None)
                    if reasoning:
                        print(reasoning, end="", flush=True)
                        final_reasoning += reasoning


                    content = delta.content
                    if content:

                        if not result and final_reasoning:
                            print("\n\n=== Reasoning complete; generating content ===\n")

                        print(content, end="", flush=True)
                        result += content
                if hasattr(chunk, 'usage') and chunk.usage:
                    usage_info = chunk.usage
                    print(f"\n✅ Token usage received: {usage_info}")
            except Exception as e:
                print(f"Error from LLM response: {e}")
                continue

            if '<think>' in result and '</think>' in result:
                result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL).strip()

            global token_cost, TOKEN_STATS

            prompt_tokens = getattr(usage_info, "prompt_tokens", None) if usage_info else None
            completion_tokens = getattr(usage_info, "completion_tokens", None) if usage_info else None


            reasoning_tokens = 0
            if usage_info is not None:
                completion_details = getattr(usage_info, "completion_tokens_details", None)
                if completion_details is not None:
                    reasoning_tokens = getattr(completion_details, "reasoning_tokens", 0) or 0

            pt = prompt_tokens or 0
            ct = completion_tokens or 0
            cr = reasoning_tokens or 0
            cv = max(ct - cr, 0)

            tt = pt + ct

            token_cost += tt


            TOKEN_STATS["total_prompt"] += pt
            TOKEN_STATS["total_completion"] += ct
            TOKEN_STATS["total_completion_reasoning"] += cr
            TOKEN_STATS["total_completion_visible"] += cv
            TOKEN_STATS["total_total"] += tt

            phase = meta.get("phase") or "unknown"
            by_phase = TOKEN_STATS["by_phase"]
            phase_stats = by_phase.setdefault(
                phase,
                {"prompt": 0, "completion": 0, "completion_reasoning": 0, "completion_visible": 0, "total": 0},
            )
            phase_stats["prompt"] += pt
            phase_stats["completion"] += ct
            phase_stats["completion_reasoning"] += cr
            phase_stats["completion_visible"] += cv
            phase_stats["total"] += tt
            return result
        except Exception as e:
            print(f"Error calling LLM: {e}")
    return {}



def call_llm_json(prompt: str, temperature: float = -1, enable_thinking: bool = False, **meta) -> dict:
    for _ in range(5):
        try:
            kwargs = _request_options(enable_thinking, json_output=True)
            if temperature != -1:
                kwargs["temperature"] = temperature

            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
            content = _get_message_content(response).strip()
            try:
                result = json.loads(content)
            except Exception as e:
                # Fallback: try to extract JSON substring.
                try:
                    json_start = content.find("{")
                    json_end = content.rfind("}") + 1
                    json_str = content[json_start:json_end]
                    result = json.loads(json_str)
                except Exception as e2:
                    try:
                        json_start = content.rfind("{")
                        json_end = content.rfind("}") + 1
                        json_str = content[json_start:json_end]
                        result = json.loads(json_str)
                    except Exception as e2:
                        print(f"Error parsing JSON from LLM response: {e}; fallback error: {e2}")
                        print(f"Response content was: {content}")
                    continue

            if result == {}:
                continue

            global token_cost, TOKEN_STATS
            pt, ct, cr = _get_usage_fields(response)
            cv = max(ct - cr, 0)
            tt = pt + ct
            token_cost += tt

            TOKEN_STATS["total_prompt"] += pt
            TOKEN_STATS["total_completion"] += ct
            TOKEN_STATS["total_completion_reasoning"] += cr
            TOKEN_STATS["total_completion_visible"] += cv
            TOKEN_STATS["total_total"] += tt

            phase = meta.get("phase") or "unknown"
            phase_stats = TOKEN_STATS["by_phase"].setdefault(
                phase,
                {"prompt": 0, "completion": 0, "completion_reasoning": 0, "completion_visible": 0, "total": 0},
            )
            phase_stats["prompt"] += pt
            phase_stats["completion"] += ct
            phase_stats["completion_reasoning"] += cr
            phase_stats["completion_visible"] += cv
            phase_stats["total"] += tt

            return result
        except Exception as e:
            print(f"Error calling LLM (json): {e}")
    return {}

def call_llm_json_stream(prompt: str, temperature: float = -1, enable_thinking: bool = False, **meta) -> dict:
    """Call LLM and parse a JSON object from response."""
    for _ in range(5):
        try:
            kwargs = _request_options(enable_thinking, json_output=True)
            if temperature != -1:
                kwargs["temperature"] = temperature

            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                stream_options={"include_usage": True},
                **kwargs
            )
            final_reasoning = ""
            final_content = ""
            usage_info = None
            try:

                for chunk in response:

                    if not chunk.choices:
                        continue

                    if hasattr(chunk, 'usage') and chunk.usage:
                        usage_info = chunk.usage
                        print(f"\n✅ Token usage received: {usage_info}")
                        continue

                    delta = chunk.choices[0].delta



                    reasoning = getattr(delta, 'reasoning_content', None)
                    if reasoning:
                        print(reasoning, end="", flush=True)
                        final_reasoning += reasoning


                    content = delta.content
                    if content:

                        if not final_content and final_reasoning:
                            print("\n\n=== Reasoning complete; generating content ===\n")

                        print(content, end="", flush=True)
                        final_content += content
                if hasattr(chunk, 'usage') and chunk.usage:
                    usage_info = chunk.usage
                    print(f"\n✅ Token usage received: {usage_info}")
            except Exception as e:
                print(f"Error parsing JSON from LLM response: {e}")
                continue

            try:
                result = json.loads(final_content)
            except Exception as e:
                try:
                    json_start = final_content.find("{")
                    json_end = final_content.rfind("}") + 1
                    json_str = final_content[json_start:json_end]
                    result = json.loads(json_str)
                except Exception as e2:
                    try:
                        json_start = final_content.rfind("{")
                        json_end = final_content.rfind("}") + 1
                        json_str = final_content[json_start:json_end]
                        result = json.loads(json_str)
                    except Exception as e2:
                        print(f"Error parsing JSON from LLM response: {e}; fallback error: {e2}")
                        print(f"Response content was: {final_content}")
                        continue

            global token_cost, TOKEN_STATS

            prompt_tokens = getattr(usage_info, "prompt_tokens", None) if usage_info else None
            completion_tokens = getattr(usage_info, "completion_tokens", None) if usage_info else None


            reasoning_tokens = 0
            if usage_info is not None:
                completion_details = getattr(usage_info, "completion_tokens_details", None)
                if completion_details is not None:
                    reasoning_tokens = getattr(completion_details, "reasoning_tokens", 0) or 0

            pt = prompt_tokens or 0
            ct = completion_tokens or 0
            cr = reasoning_tokens or 0
            cv = max(ct - cr, 0)

            tt = pt + ct

            token_cost += tt


            TOKEN_STATS["total_prompt"] += pt
            TOKEN_STATS["total_completion"] += ct
            TOKEN_STATS["total_completion_reasoning"] += cr
            TOKEN_STATS["total_completion_visible"] += cv
            TOKEN_STATS["total_total"] += tt

            phase = meta.get("phase") or "unknown"
            by_phase = TOKEN_STATS["by_phase"]
            phase_stats = by_phase.setdefault(
                phase,
                {"prompt": 0, "completion": 0, "completion_reasoning": 0, "completion_visible": 0, "total": 0},
            )
            phase_stats["prompt"] += pt
            phase_stats["completion"] += ct
            phase_stats["completion_reasoning"] += cr
            phase_stats["completion_visible"] += cv
            phase_stats["total"] += tt
            return result
        except Exception as e:
            print(f"Error calling LLM: {e}")
    return {}




class Agent:
    def __init__(self, name: str, id: int, schedule: dict, memory: dict | None = None):
        self.name = name
        self.id = id
        self.schedule = schedule
        self.memory = memory or {}
        self.day_schedules = {d : [] for d in range(1, schedule.get("total_days", 1) + 1)}
        for activity in self.schedule.get("activities", []):
            day = activity.get("date")
            self.day_schedules[day].append(activity)


    def process_schedule(self, schedule: dict) -> dict:
        new_schedule = {"time_unit": schedule.get("time_unit"), "total_days": schedule.get("total_days"), "activities": []}
        for activity in schedule.get("activities", []):

            activity_copy = {
                "date": activity.get("date"),
                "duration": activity.get("duration"),
                "description": activity.get("description")
            }
            new_schedule["activities"].append(activity_copy)
        return new_schedule

    def speak(
        self,
        task_prompt: str,
        conversation_history: list[str],
        team_memory: dict,
        leader_instruction: str,
        *,
        style: str,
        max_rounds: int,
        current_round: int,
        use_memory: bool,
    ):
        memory_data = self.memory.get("summary_data") if isinstance(self.memory, dict) else None
        memory_snapshot = json.dumps(memory_data if memory_data is not None else (self.memory or {}), ensure_ascii=False)
        schedule_view = (
            {"time_unit": self.schedule.get("time_unit"), "activities": self.schedule.get("activities", [])}
            if isinstance(self.schedule, dict)
            else (self.schedule or {})
        )
        schedule_snapshot = json.dumps(schedule_view, ensure_ascii=False)
        rounds_info = f"Rounds: {current_round}/{max_rounds}.\n"
        conversation_text = "".join(conversation_history or [])

        if style == "protocol":
            duration_minutes = int((team_memory or {}).get("slot_length_minutes") or 20)
            memory_view = json.dumps(team_memory or {}, ensure_ascii=False)
            personal_block = f"Your Personal Memory (available_slots summary_data):\n{memory_snapshot}\n" if use_memory else ""


            is_discuss_phase = team_memory.get("phase") == "DISCUSS" or leader_instruction.lower().find("phase: discuss") != -1
            is_voting_phase = team_memory.get("phase") == "VOTE" or leader_instruction.lower().find("phase: vote") != -1

            if is_discuss_phase:
                prompt = f"""
                    You are Agent #{self.id} ("{self.name}").
                    **TASK: BROAD AVAILABILITY SCAN (PHASE 0)**

                    ### 1. GOAL
                    Identify your **TOP 3** best time windows (Standard Blocks) where you have the **HIGHEST ACCUMULATED FREE TIME**.

                    ### 2. YOUR TRUTH (WHITE LIST)
                    **You are ONLY free strictly inside these intervals:**
                    {personal_block}
                    *(Note: All slots listed here are already valid. Do not filter them further.)*

                    ### 3. SCANNING LOGIC (Step-by-Step)

                    **STEP A: DEFINE STANDARD BLOCKS**
                    Map your free slots into these 3 fixed windows for each day (Day 1-5):
                    1. **Morning:** 08:00 - 12:00
                    2. **Afternoon:** 14:00 - 18:00
                    3. **Evening:** 20:00 - 24:00

                    **STEP B: CALCULATE SCORE (Pure Summation)**
                    Since your memory is already pre-filtered, simply calculate the **Total Volume** of free time.
                    * **Math:** For every free slot inside a block, calculate `(End_Time - Start_Time)` in minutes and sum them up.
                    * *Example:* If you have `08:00-08:30` (30m) and `11:00-12:00` (60m) in Morning -> **Score = 90**.

                    **STEP C: RANK & SELECT (CRITICAL FORMATTING)**
                    1. **Sort:** Rank all blocks by Total Score (High to Low).
                    2. **Select:** Pick the Top 3 winners.
                    3. **Map Back:** Convert the winner back to its **Day ID**.

                    **⚠️ DATA TYPE WARNING (Read Carefully):**
                    * **CORRECT First Element:** The **Day ID** (e.g., 1, 2, 3, 4, 5).
                    * **WRONG First Element:** The calculated minutes (e.g., 90, 120). **NEVER output the score number.**

                    ### 4. OUTPUT FORMAT
                    Return JSON ONLY.
                    Structure: `[Day_Integer, "Block_Name"]`

                    {{
                    "broad": [
                        [2, "Afternoon"],  // Correct: Day 2. (NOT 240)
                        [1, "Morning"],    // Correct: Day 1. (NOT 90)
                        [3, "Evening"]     // Correct: Day 3.
                    ]
                    }}
                    """

            elif is_voting_phase:

                prompt = f"""
                    You are Agent #{self.id} ("{self.name}").
                    **TASK: BATCH AVAILABILITY CHECK (PHASE 2)**

                    ### 1. INPUT (CANDIDATE SLOTS)
                    The Leader needs you to vote on the following list.
                    {leader_instruction}

                    ### 2. YOUR TRUTH (WHITE LIST)
                    **You are ONLY free strictly inside these intervals:**
                    {personal_block}

                    ### 3. THINKING PROCESS INSTRUCTIONS (Strict Execution)
                    **You must perform a detailed "Mental Simulation" before outputting JSON.**
                    **Do NOT skip any steps. Write this out in your thought block:**

                    1.  **Initialize:** Set `votes = []`.
                    2.  **Loop:** Iterate through EVERY slot in the Input list (Index `i` from 0 to Last).
                        * **Context:** "Checking Slot `i`: Day `D` `Start`-`End`".
                        * **Fetch:** Look up your White List for Day `D`. (If Day `D` is missing, result is 0).
                        * **Compare:** Is there an interval `Free_Start`-`Free_End` where:
                            * `Free_Start` <= `Start`
                            * `End` <= `Free_End`
                        * **Log:** Write down "Slot `i`: Matches `Free_Interval`? -> YES/NO".
                        * **Append:** Add 1 or 0 to `votes`.
                    3.  **Finalize:** Count the length of `votes`. Does it match the input size?

                    **Example Thought Trace (Do this for REAL inputs):**
                    * "Checking Slot 0: Day 1 08:00-08:20. My Day 1 has 08:00-09:00. 08:00<=08:00 and 08:20<=09:00. MATCH. Vote 1."
                    * "Checking Slot 1: Day 1 09:40-10:00. My Day 1 has 08:00-09:00. No match. Vote 0."
                    * ... (Repeat for all 15 items)

                    ### 4. OUTPUT FORMAT
                    Return JSON ONLY.
                    {{
                    "target_count": 15, // The exact number of input items
                    "math_verified": true,
                    "v": [1, 0, 1, ...] // Copy your mental `votes` list here.
                    }}
                    """

            else:
                prompt = f"""
                    You are Agent #{self.id} ("{self.name}").
                    **TASK: PROPOSE 2-4 ANCHOR SLOTS (PHASE 1)**

                    ### 1. CONTEXT
                    **Target Duration:** {duration_minutes} minutes (EXACTLY).
                    **Leader Instruction:** {leader_instruction}
                    **Your Schedule:** {personal_block}
                    **Dead Zones (Rejected):** {memory_view}

                    **>>> PEERS' RECENT PROPOSALS:**
                    {conversation_text}

                    ### 2. STANDARD TIME DEFINITIONS
                    Use these boundaries:
                    1. **Morning:** 08:00 - 12:00
                    2. **Afternoon:** 14:00 - 18:00
                    3. **Evening:** 20:00 - 24:00

                    ### 3. EXECUTION ALGORITHM (STRICT SEQUENCE)
                    **STEP A: DEFINE SEARCH SCOPE**
                    * **Zone Lock:** Strictly follow Leader's Priority Zones (e.g., Day 3 Afternoon = 14:00-18:00).
                    * **Quotas:** If asked for 2 slots in Priority #1, you MUST provide 2 distinct slots there.

                    **STEP B: GENERATE CANDIDATES (Self-Validation)**
                    * **Rule:** You must be 100% FREE. Slice your availability into {duration_minutes}-min chunks.

                    **STEP C: GAP ANALYSIS (CRITICAL - FILL THE VOID)**
                    * **Analyze Peers:** Look at `PEERS' RECENT PROPOSALS`.
                    * **Hour Buckets:** Categorize existing proposals into hours (e.g., 14h, 15h, 16h, 17h).
                    * **Identify Gaps:** Find which hours have **ZERO or FEW** proposals.
                        * *Scenario 1:* Peers picked 14:xx and 17:xx. -> **GAP is 15:xx and 16:xx**.
                        * *Scenario 2:* Peers picked 14:xx and 15:xx. -> **GAP is 16:xx and 17:xx**.
                        * *Scenario 3 (First Mover):* If no proposals exist, assume all hours are Gaps.

                    **STEP D: STRATEGIC SELECTION**
                    * **Primary Logic:** You MUST select slots from the **GAPS** identified in Step C.
                        * *Instruction:* If 15:xx is empty, your TOP PRIORITY is to propose a slot in 15:xx. Do not pile onto 14:xx if 15:xx is empty.
                    * **Secondary Logic:** If all hours are covered, try to pick a specific time *between* existing slots (e.g., if 14:00 and 14:40 are taken, try 14:20).
                    * **Result:** Your 2 proposals should ideally fill two different empty buckets (e.g., one in 15h, one in 16h).

                    **STEP E: SANITY CHECK**
                    1. Am I free? (Yes)
                    2. Is duration exactly {duration_minutes}? (Yes)
                    3. Is it inside the requested Zone? (Yes)

                    ### 4. OUTPUT FORMAT
                    Return JSON ONLY. Exactly 2-4 slots.
                    {{"p": [[day_int, "HH:MM", "HH:MM"], ...]}}
                """

            return call_llm_json_stream(
                prompt,
                tag="agent_speak",
                role="agent",
                phase="agent_speak",
                task_type="meeting",
                style=style,
                round=current_round,
                agent_id=self.id,
                enable_thinking=True,
            )

        candidate_slots = (team_memory or {}).get("candidate_slots", [])
        rejected_slots = (team_memory or {}).get("rejected_slots", [])

        rejected_view = rejected_slots[-60:] if isinstance(rejected_slots, list) and len(rejected_slots) > 60 else rejected_slots

        shared_snapshot = json.dumps({"candidate_slots": candidate_slots, "rejected_slots": rejected_view}, ensure_ascii=False)

        if use_memory:
            personal_block = f"**Your VALID FREE INTERVALS (White List):**\n{memory_snapshot}\n*(You are ONLY free inside these specific intervals)*"
            team_block = f"Team memory (reference only):\n{shared_snapshot}\n"

            instruction_block = f"""
            ### [INSTRUCTIONS: SMART STRATEGY]

            1. **STEP 1: IDENTIFY TARGETS (The "One-Strike" Filter)**
               - **Scan Scope:** Look at the Leader's list AND recent peer proposals.
               - **Filter Dead Slots:** If *ANYONE* (including you previously) rejected a slot, it is **DEAD**. Ignore it.
               - **Identify Active Slots:** Any slot with ZERO rejections is **ALIVE**.
               - **Mandatory Duty:** You **MUST** vote on every ALIVE slot first.

            2. **STEP 2: STRICT INTERNAL CHECK (White List)**
               - **Rule:** Compare each *ALIVE* slot against your **White List**.
               - **Constraint:** The slot must fit **ENTIRELY** inside one of your Free Intervals.
               - *Note:* Precision matters. 10:05 end != 10:00 end.

            3. **STEP 3: EXECUTE RESPONSE (Strict Format)**
               - **Part A - Voting (Required):** For EACH Alive Slot:
                 `For [Day X HH:MM-HH:MM]: [Yes/No]`
               - **Part B - Proposing (Conditional):** If (and ONLY if) you rejected the Alive slots or there were none, propose **2 NEW distinct slots**.
                 `Proposing: [Day X HH:MM-HH:MM], [Day Y HH:MM-HH:MM]`

               - **STRATEGY: THE "DEEP POCKET" RULE**
                 - **Do NOT pick random slots.**
                 - **Look at your White List:** Identify your **LONGEST** continuous free blocks (e.g., a 2-hour window vs. a 20-min window).
                 - **Action:** Generate your 2 proposals strictly from these **longest** blocks.
                 - **Cluster:** It is BETTER to propose two slots close together (e.g., "14:00-14:20" and "14:20-14:40") from a large block, rather than scattering them. This maximizes the chance of hitting the team's intersection.

            4. **PRIVACY:**
               - Only share specific decisions. Do not reveal full agenda.
            """

        else:
            personal_block = f"**Your SCHEDULED ACTIVITIES (Busy/Black List):**\n{schedule_snapshot}\n*(You are BUSY during these times. Implicitly free otherwise.)*"
            team_block = ""

            instruction_block = f"""
            ### [INSTRUCTIONS]
            1. **CONFLICT CHECK (No Overlap):**
               - You see a list of **Busy Activities**.
               - To accept a slot, ensure it does **NOT overlap** (even by 1 minute) with any activity in your schedule.
               - Also ensure it does not violate Global Constraints (Sleep/Meals).

            2.**PROPOSAL FORMAT (Strict):**
               - When proposing, you MUST provide the specific **Day ID** and **Time Range**.
               - **Rule:** Use "Day X" (e.g., "Day 1"), **NOT** weekday names (like "Monday").
               - *Wrong:* "How about Monday morning?" or "I am free at 10:00".

            3. **PRIVACY:**
               - Do NOT reveal your full agenda. Only share specific decisions (e.g., "I am free at 10:00").
            """


        prompt = f"""
            You are Agent #{self.id} ("{self.name}"), participating in a collaborative meeting scheduling task.
            Your goal is to negotiate with your team to find ONE common time slot that fits everyone's schedule.

            ### [SYSTEM STATUS]
            {rounds_info}

            ### [TASK DIRECTIVE]
            {task_prompt}

            ### [YOUR PRIVATE KNOWLEDGE]
            {personal_block}
            (NOTE: This schedule is PRIVATE. Do not copy-paste it into the chat. Only you can see this.)

            ### [TEAM CONTEXT]
            Participants: {team_block}
            Conversation History:
            {conversation_text}

            {instruction_block}

            ### [OUTPUT FORMAT]
            Output your response in plain text.
            If the group has reached a final agreement, strictly follow the format defined in the Task Directive.
            """
        return call_llm(
            prompt,
            tag="agent_speak_free",
            role="agent",
            phase="agent_speak",
            task_type="meeting",
            style=style,
            round=current_round,
            agent_id=self.id,
            enable_thinking=True,
        )

class LeaderAgent:
    def __init__(self, name: str = "Leader", style: str = "protocol"):
        self.name = name
        self.style = style

    def initiate_task(self, members: list[str], duration: int, max_turns: int, days: int) -> str:
        member_names = ", ".join(members)
        return f"""Goal: Find a single common time slot of exactly {duration} minutes for the following participants: {member_names}.
            Common unavailable times for all participants:
            - 00:00 - 07:00 (Sleep)
            - 07:00 - 08:00 (Breakfast)
            - 12:00 - 14:00 (Lunch)
            - 18:00 - 20:00 (Dinner)

            Constraints:
            1. Privacy: Individual schedules are private. Participants must propose specific slots without revealing full daily agendas.
            2. Efficiency: The group must reach a final agreement within {max_turns} turns, proposals outside Day 1-{days} are INVALID.
            """

    @staticmethod
    def _parse_day_slot(text: str) -> dict | None:
        if not isinstance(text, str) or not text.strip():
            return None
        m = re.search(
            r"(?:\bDay\s*)?(\d+)\s*(?:,|\s)+([0-2]?\d:\d{2})\s*[-–—~]\s*([0-2]?\d:\d{2})",
            text,
            flags=re.IGNORECASE,
        )
        if not m:
            return None
        try:
            day = int(m.group(1))
        except Exception:
            return None
        return {"day": day, "start_time": m.group(2), "end_time": m.group(3)}

    def check_completion(
        self,
        conversation_history: list[str],
        task_prompt: str,
        team_memory: dict,
        max_rounds: int,
        current_round: int,
        last_leader_instruction: str,
    ):
        team_mem_str = json.dumps(team_memory or {}, ensure_ascii=False)
        turn_log = json.dumps(conversation_history or [], ensure_ascii=False)
        duration_minutes = int(team_memory.get("slot_length_minutes") or 20)

        rounds_info = (
            f"Total allowed rounds: {max_rounds}. This is round {current_round}. "
            "Prefer decisions that move quickly toward a final slot.\n"
        )

        if self.style == "protocol":
            is_voting_phase = team_memory.get("phase") == "VOTE" or last_leader_instruction.lower().find("phase: vote") != -1
            is_discuss_phase = team_memory.get("phase") == "DISCUSS" or last_leader_instruction.lower().find("phase: discuss") != -1
            if is_voting_phase:
                prompt = f"""
                    You are the **Meeting Adjudicator** (Phase: VOTE).
                    **Goal:** Strict verification of UNANIMOUS CONSENSUS (100% Agreement).

                    ### 1. INPUTS
                    **The Candidates:** (Refer to the numbered list in the Leader's last message, e.g., "0: Day 1 10:00...")
                    **The Votes:** {turn_log}

                    ### 2. STRICT JUDGMENT LOGIC (Step-by-Step)
                    1. **Vertical Summation:** For EACH Candidate Index `i`, sum the votes from ALL 5 agents.
                    - `Sum[i] = A1[i] + A2[i] + A3[i] + A4[i] + A5[i]`
                    2. **The Threshold:**
                    - **WIN:** ONLY if `Sum[i] == 5`.
                    - **FAIL:** If `Sum[i] < 5` (e.g., 4, 3, 2, 1, 0).
                    - *CRITICAL:* 4 votes is a FAIL. Do not round up.
                    3. **Slot Retrieval (CRITICAL):**
                    - If a winner is found at Index `k`, you MUST look back at the Candidate List and **extract the exact text string** for that index (e.g., "Day 3 09:20-09:40").

                    ### 3. OUTPUT FORMAT
                    Return JSON ONLY.
                    **MANDATORY:**
                    1. You MUST output `vote_tally` to prove you counted.
                    2. In Case A, `conclusion` MUST include the **Specific Slot String**.

                    **Case A: Winner Found (Sum == 5)**
                    {{
                    "vote_tally": [3, 5, 2, 4, ...],  // LIST ALL SUMS
                    "is_complete": true,
                    "conclusion": "Winner found at Index X (5/5 votes). Slot: Day <d> <HH:MM>-<HH:MM>",
                    "next_instruction": "Meeting Finalized."
                    }}

                    **Case B: No Winner (All Sums < 5)**
                    {{
                    "vote_tally": [3, 4, 2, 4, ...], // LIST ALL SUMS
                    "is_complete": false,
                    "conclusion": "No consensus. Best candidate received max X/5 votes.",
                    "next_instruction": "PHASE: COLLECT. Previous candidates rejected. Propose distinct NEW slots distributed across different days."
                    }}
                    """
            elif is_discuss_phase:
                # Leader Phase 0 Logic (Aggregation & Ranking)
                prompt = f"""
                    You are the **Meeting Strategist** (Phase: BROAD SCAN).
                    **Goal:** ACCURATELY count the votes and identify the **TOP 2** overlaps.

                    ### 1. INPUTS
                    **Agents' Broad Availability:**
                    {turn_log}

                    ### 2. EXECUTION PROTOCOL (Strict "Map-Reduce" Logic)
                    You must generate a **Plain Text Analysis** before the JSON. Do not skip steps.

                    **STEP 1: RAW EXTRACTION (The "Map" Phase)**
                    - Read the input line by line.
                    - **WRITE DOWN** every single `[Day, Block]` pair found. Do not group them yet.
                    - *Format:* "Member 1 voted: D1-Morning, D5-Evening..."

                    **STEP 2: AGGREGATION (The "Reduce" Phase)**
                    - Now, look at your list from Step 1. Group identical blocks and count them.
                    - **List ALL blocks with > 1 vote.**
                    - *Format:* "Day X [Block]: N votes"

                    **STEP 3: SORTING & RANKING (The Logic Check)**
                    - **Rule 1 (Votes):** 3 votes > 2 votes. (CRITICAL: Do not pick a 2-vote block if a 3-vote block exists!)
                    - **Rule 2 (Tie-Breaker):** If votes are equal, Earlier Day wins. If Days are equal, Earlier Time wins.
                    - *Comparison:* Compare your top candidates. Who wins? Who is second?

                    **STEP 4: GENERATE JSON**
                    - Transfer the verified Top 1 and Top 2 into the output.

                    ### 3. OUTPUT FORMAT
                    **Part 1: Analysis (Plain Text)**
                    [Perform Step 1, 2, and 3 here. Show your work.]

                    **Part 2: Result (JSON)**
                    {{
                    "is_complete": false, // FORCE THIS TO FALSE
                    "conclusion": "Ranking: [Winner] (N votes), [Runner-up] (M votes).",
                    "next_instruction": "PHASE: COLLECT.\n**STRATEGY: PRIORITY #1 [Insert Winner] & PRIORITY #2 [Insert Runner-up].**\n**STRICT PROPOSAL QUOTA:**\n1. **Priority #1 (Main):** You MUST propose **2 slots** strictly inside this zone.\n2. **Priority #2 (Backup):** Propose exactly **1 slot** inside this zone.\n**Total:** 3 slots per agent."
                    }}
                    """

            else:
                prompt = f"""
                    You are the **Meeting Protocol Coordinator** (Phase: COLLECT).
                    **Goal:** Extract, Clean, Sort, and List valid slot proposals for voting.

                    ### 1. INPUTS
                    - **Raw Proposals:** {turn_log}
                    - **Blacklist (Previously Rejected):** {team_mem_str}
                    - **Target Duration:** {duration_minutes} minutes.

                    ### 2. CLEANING RULES (Execute in Order)

                    1. **Extraction:** Parse all proposed slots `[Day, Start, End]` from the raw log.
                    2. **Duration Check:** Discard slots where `(End - Start) != {duration_minutes}` mins.
                    3. **Blacklist Check:** Discard any slot that EXACTLY matches an entry in the `Blacklist`.

                    4. **Deduplication (STRICT):**
                    - Remove **EXACT DUPLICATES** only (Same Day + Same Start + Same End).
                    - **KEEP OVERLAPS:** If Slot A is `09:00-09:20` and Slot B is `09:10-09:30`, **KEEP BOTH**. They are different options.

                    5. **Sorting (Chronological):**
                    - Sort the final list by **Day (Ascending)**, then by **Start Time (Ascending)**.
                    - Example order: Day 1 09:00, Day 1 09:30, Day 2 08:00...

                    ### 3. OUTPUT FORMAT
                    Return JSON ONLY.
                    - Format the `next_instruction` as a clear, numbered list for the next phase.

                    {{
                    "is_complete": false,
                    "conclusion": "Collected X unique valid proposals (sorted).",
                    "next_instruction": "PHASE: VOTE. Check availability (1=Yes, 0=No) for:\\n0: Day 1 09:00-09:20\\n1: Day 1 09:10-09:30\\n2: Day 2 14:00-14:20\\n..."
                    }}
                    """
        else:
            prompt = f"""
            You are the **Meeting Adjudicator**.
            **Role:** You are the "Status Board" of the meeting. You do NOT make suggestions. You ONLY report the status of slots.

            ### [INPUT DATA]
            {rounds_info}
            **Conversation Log (Current Round):**
            {turn_log}
            **Rejected History:** {team_mem_str}

            ### [STATUS CLASSIFICATION LOGIC] (Execute Mentally)

            **1. IDENTIFY REJECTED (The Red List):**
            - Scan the `turn_log` for ANY "No" or "Reject".
            - Identify which slot was being rejected (often the one from the previous instruction OR a specific new proposal).
            - Add these to the "Rejected List".

            **2. IDENTIFY ACTIVE/PENDING (The Green List):**
            - Scan the `turn_log` for **NEW PROPOSALS** made in *this current round*.
            - Check: Has this new proposal received a "No" *in this same round*?
            - **YES:** Move to Rejected List.
            - **NO (or Silence):** Mark as **"PENDING"**. (These are the slots agents need to discuss next).

            **3. CHECK CONSENSUS:**
            - Did any Pending slot receive "Yes" from **ALL 5** agents? -> `is_complete: true`, extract that slot as the final agreed time. Conclusion must specify it, e.g., "Final slot agreed: Day 3 10:00-10:20".

            ### [OUTPUT FORMAT]
            Return JSON ONLY.

            {{
            "is_complete": boolean,
            "conclusion": "Brief summary (e.g., 'Old candidates rejected. 5 new slots proposed.'), if find a final slot, specify it here (e.g., 'Final slot agreed: Day 3 10:00-10:20').",
            "next_instruction": "REJECTED: [List slots rejected in this round].\nPENDING: [List slots proposed this round that are NOT rejected yet].\nACTION: Vote on PENDING slots or propose NEW ones."
            }}

            **Constraint for `next_instruction`:**
            - **CRITICAL:** You MUST list the **PENDING** slots found in the current log. If you ignore them, the agents will forget them.
            - **Example:** "REJECTED: Day 5 14:40. PENDING: Day 1 14:00, Day 4 16:00. ACTION: Agents, please vote on the Pending slots."
            """

        for _ in range(3):
            result = call_llm_json_stream(
                prompt,
                tag="leader_check",
                role="leader",
                phase="leader_check",
                task_type="meeting",
                style=self.style,
                round=current_round,
                enable_thinking=True,
            ) or {}
            is_complete = bool(result.get("is_complete", False))
            conclusion = str(result.get("conclusion") or "")
            next_instruction = str(result.get("next_instruction") or "")
            if result != {} and conclusion != "" and  next_instruction != "":
                break

        return is_complete, conclusion, next_instruction


class Orchestrator:
    def __init__(self, schedule_dir: str, *, style: str, use_memory: bool):
        self.schedule_dir = os.path.abspath(schedule_dir)
        self.schedules = self._load_schedules(self.schedule_dir)
        if not self.schedules:
            raise ValueError("Could not load schedule data or the folder is empty.")

        self.agents = [Agent(s["member_name"], s["member_id"], s) for s in self.schedules]
        self.agents.sort(key=lambda a: a.id)
        self.member_names = [a.name for a in self.agents]
        self.member_ids = [a.id for a in self.agents]
        self.days = self.schedules[0].get("total_days", 10)

        self.style = style
        self.use_memory = use_memory
        self.leader = LeaderAgent(style=style)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.logs_dir = ensure_logs_dir(base_dir)
        self.log_file_path: str | None = None
        self.meta_file_path: str | None = None
        self.run_start_time: str | None = None

        self.team_memory: dict = {}
        self.conversation_history: list[str] = []
        self.last_leader_instruction: str = ""

    def _load_schedules(self, dir_path: str) -> list:
        schedules = []
        pattern = os.path.join(dir_path, "Member*_final_schedule.json")
        files = glob.glob(pattern)
        for file_path in files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    schedules.append(json.load(f))
            except FileNotFoundError:
                print(f"Error: Schedule file {file_path} not found.")
                return []
            except json.JSONDecodeError:
                print(f"Error: Schedule file {file_path} is not correctly formatted.")
                return []
        return schedules

    @staticmethod
    def _hhmm_to_minutes(hhmm: str) -> int | None:
        if not isinstance(hhmm, str):
            return None
        m = re.match(r"^([0-2]?\d):(\d{2})$", hhmm.strip())
        if not m:
            return None
        h = int(m.group(1))
        mi = int(m.group(2))
        if h < 0 or h > 23 or mi < 0 or mi > 59:
            return None
        return h * 60 + mi

    @staticmethod
    def _minutes_to_hhmm(minutes: int) -> str:
        h = max(min(int(minutes) // 60, 23), 0)
        m = max(min(int(minutes) % 60, 59), 0)
        return f"{h:02d}:{m:02d}"


    def _initialize_memories(self, duration_minutes: int, mode: str = "serial"):
        if self.use_memory:
            target_days = [d for d in range(1, self.days + 1)]
            init_mode = mode.lower()

            def call_agent_memory(agent: Agent, target_day: int | None = None) -> object:
                length = 0
                for _ in range(3):
                    agent_prompt = self._generate_agent_memory_prompt(agent, duration_minutes=duration_minutes, target_day=target_day)
                    if model_name.startswith("qwen") or model_name.startswith("deepseek") or model_name.startswith("gemini") or model_name.startswith("gpt"):
                        memory_response = call_llm_json_stream(
                            agent_prompt,
                            tag="init_memory",
                            role="agent",
                            phase="init_memory",
                            task_type="meeting",
                            style=self.style,
                            agent_id=agent.id,
                            temperature=0.1
                        )
                    else:
                        memory_response = call_llm_json(
                            agent_prompt,
                            tag="init_memory",
                            role="agent",
                            phase="init_memory",
                            task_type="meeting",
                            style=self.style,
                            agent_id=agent.id,
                            temperature=0
                        )
                    free_slot = memory_response["all_gaps"]
                    extract_prompt =  self._generate_agent_extract_prompt(agent, duration_minutes=duration_minutes, free_slots=free_slot)
                    if model_name.startswith("qwen") or model_name.startswith("deepseek") or model_name.startswith("gemini") or model_name.startswith("gpt"):
                        response = call_llm_json_stream(
                            extract_prompt,
                            tag="init_memory",
                            role="agent",
                            phase="init_memory",
                            task_type="meeting",
                            style=self.style,
                            agent_id=agent.id,
                        )
                    else:
                        response = call_llm_json(
                            extract_prompt,
                            tag="init_memory",
                            role="agent",
                            phase="init_memory",
                            task_type="meeting",
                            style=self.style,
                            agent_id=agent.id,
                        )
                    memory_response['free_slots'] =  response["valid_slots"]
                    if memory_response['free_slots'] != [] and len(memory_response['free_slots']) > 0:
                        return  memory_response
                return []

            # Version A: SERIAL (agents x days in nested loops)
            def init_serial():
                for agent in self.agents:
                    merged: dict[str, list[str]] = {}
                    for d in target_days:
                        try:
                            memory_response = call_agent_memory(agent, d)
                            merged[str(d)] = memory_response['free_slots'] if isinstance(memory_response, dict) and 'free_slots' in memory_response else []
                        except Exception as e:
                            print(f"Agent {agent.id} memory initialization failed (day {d}): {e}")
                            merged[str(d)] = []
                    agent.memory = {
                        "agent_id": agent.id,
                        "summary_type": "available_slots",
                        "summary_data": merged,
                    }

            # Version B: PARALLEL (flatten (agent, day) tasks into a single pool)
            def init_parallel():
                merged_by_agent: dict[int, dict[str, list[str]]] = {a.id: {} for a in self.agents}

                def process_agent_day(agent: Agent, d: int) -> tuple[int, int, list[str]]:
                    try:
                        memory_response = call_agent_memory(agent, d)
                        return agent.id, d, memory_response['free_slots'] if isinstance(memory_response, dict) and 'free_slots' in memory_response else []
                    except Exception as e:
                        print(f"Agent {agent.id} memory initialization failed (day {d}): {e}")
                        return agent.id, d, []

                max_workers = min(len(self.agents), 5)

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_key = {
                        executor.submit(process_agent_day, agent, d): (agent.id, d)
                        for agent in self.agents
                        for d in target_days
                    }
                    for future in as_completed(future_to_key):
                        agent_id, d = future_to_key[future]
                        try:
                            _agent_id, _d, slots = future.result()
                            merged_by_agent[agent_id][str(d)] = slots
                        except Exception as e:
                            print(f"Task failed with error: {e}")
                            merged_by_agent[agent_id][str(d)] = []

                for agent in self.agents:
                    merged_for_agent = merged_by_agent.get(agent.id, {})
                    ordered = {str(d): merged_for_agent.get(str(d), []) for d in target_days}
                    agent.memory = {
                        "agent_id": agent.id,
                        "summary_type": "available_slots",
                        "summary_data": ordered,
                    }

            if init_mode in {"serial", "s"}:
                init_serial()
            else:
                init_parallel()

        self.team_memory = {
            "task_type": "meeting",
            "slot_length_minutes": duration_minutes,
            "global_rules": {
                "Sleep": "00:00-07:00",
                "Breakfast": "07:00-08:00",
                "Lunch": "12:00-14:00",
                "Dinner": "18:00-20:00",
            },
            "agents": self.member_ids,
            "days": [list(range(1, self.days + 1))],
            "candidate_slots": [],
            "rejected_slots": [],
        }
        if self.style == "protocol":
            self.team_memory["phase"] = "DISCUSS"
            self.team_memory["broad_time_blocks"] = []

    def _global_rules_text(self) -> str:
        return (
            "The following times are UNAVAILABLE for everyone: "
            "Sleep 00:00-07:00; Breakfast 07:00-08:00; Lunch 12:00-14:00; Dinner 18:00-20:00"
        )

    def _generate_agent_memory_prompt(self, agent: Agent, *, duration_minutes: int, target_day: int | None = None) -> str:
        if target_day is not None:
            schedule_json = json.dumps(agent.day_schedules.get(target_day, {}), ensure_ascii=False)
        else:
            schedule_json = json.dumps(agent.schedule, ensure_ascii=False)
        duration = int(duration_minutes)
        global_rules = self._global_rules_text()

        agent_prompt = f"""
            You are a **Timeline Subtraction Engine**.
            **Task:** Calculate free time gaps by subtracting busy blocks from the full 24-hour day ({target_day}).

            ### 1. CONFIGURATION
            - **Target Day:** {target_day}
            - **Day Boundaries:** **00:00 to 24:00** (Strict).
            - **Output Mode:** **RAW DUMP**. Do NOT filter.
            - **Critical Rule:** You must account for the time **AFTER** the last scheduled event up to midnight.

            ### 2. INPUT DATA
            **Global Blocked Rules:**
            {global_rules}

            **Personal Busy Schedule (Day {target_day}):**
            {schedule_json}

            ### 3. EXECUTION ALGORITHM (The "Sculptor" Method)
            Imagine a solid block of time from **00:00 to 24:00**. You must "chisel away" the busy parts.

            **STEP 1: COLLECT & SORT ALL OBSTACLES**
            1.  **Global:** Include Sleep (e.g., 00:00-07:00), Meals, etc.
            2.  **Personal:** Include EVERY entry from the schedule list (Work, Study, Rest, etc.).
            3.  **Sort:** Arrange all these blocked times chronologically by Start Time.

            **STEP 2: MERGE OBSTACLES (Crucial)**
            * If Block A ends at 10:00 and Block B starts at 10:00 (or overlaps), **FUSE** them into one giant block.
            * *Example:* Dinner (18:00-20:00) + Study (21:00-21:10) -> Are they connected? No.
            * *Gap Check:* Is there space between 20:00 and 21:00? **YES**.
            * *Tail Check:* Is there space between 21:10 and 24:00? **YES**.

            **STEP 3: EXTRACT THE GAPS**
            * **Head Gap:** From Day Start (e.g., 07:00) to First Event.
            * **Body Gaps:** Between merged busy blocks.
            * **Tail Gap (OFTEN MISSED):** From the **Last Busy End Time** until **24:00** (or next Sleep start).

            ### 4. OUTPUT FORMAT
            Return JSON ONLY.
            {{
            "day": "{target_day}",
            "all_gaps": [
                "HH:MM-HH:MM",
                ...
            ]
            }}
            """
        return agent_prompt

    def _generate_agent_extract_prompt(self, agent: Agent, *, duration_minutes: int, free_slots: list[str]) -> str:

        slots_json = json.dumps(free_slots, ensure_ascii=False)
        duration = int(duration_minutes)

        agent_prompt = f"""
        You are a **Strict Duration Auditor**.
        **Task:** Filter the provided list of "Candidate Slots" and remove ANY slot strictly shorter than {duration} minutes.

        ### 1. CONFIGURATION
        - **Minimum Required Duration:** {duration} minutes.
        - **Input Data:** A list of raw time gaps identified from a schedule.
        - **Objective:** Keep valid slots. Discard invalid (too short) slots.

        ### 2. INPUT DATA (Candidate Slots)
        {slots_json}

        ### 3. VALIDATION LOGIC (Mental Math)
        For **EACH** slot in the input list, you must verify the duration:
        1. **Calculate:** `Duration_Minutes = End_Time - Start_Time`
        2. **Compare:** - IF `Duration_Minutes` < {duration}: **REJECT** (Too short).
           - IF `Duration_Minutes` >= {duration}: **KEEP**.

        **Example (Min={duration}):**
        - Slot "09:00-09:05" (5m)  -> **REJECT**
        - Slot "14:00-14:{duration}" ({duration}m) -> **KEEP**
        - Slot "16:00-18:00" (120m) -> **KEEP**

        ### 4. OUTPUT FORMAT
        Return JSON ONLY.
        **MANDATORY:** You must populate the `audit_log` first to prove you calculated the math.
        {{
            "audit_log": [
                "09:00-09:10 (10m) -> REJECT (< {duration}m)",
                "12:00-14:00 (120m) -> KEEP (>= {duration}m)",
                ...
            ],
            "valid_slots": [
                "12:00-14:00",
                ...
            ]
        }}
        """
        return agent_prompt

    def _init_logger(self, task_prompt: str):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_root = os.path.join(self.logs_dir, "meeting")
        os.makedirs(run_root, exist_ok=True)
        run_dir = os.path.join(run_root, f"{timestamp}_{model_name}")
        os.makedirs(run_dir, exist_ok=True)

        self.log_file_path = os.path.join(run_dir, "log.txt")
        self.meta_file_path = os.path.join(run_dir, "meta.json")
        self.run_start_time = datetime.now().isoformat()

        header_lines = [
            f"Start Time: {self.run_start_time}",
            "Task Type: meeting",
            f"Style: {self.style}",
            f"Scenario Dir: {self.schedule_dir}",
            f"Members: {', '.join(self.member_names)}",
            "",
            f"Initial Task Prompt by {self.leader.name}:",
            task_prompt,
            "",
        ]
        with open(self.log_file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(header_lines))

        self._log(f"memory initialized token cost: {token_cost}")
        for agent in self.agents:
            self._log(f"Agent{agent.id} Memory:")
            self._log_json(agent.memory)

    def _log(self, line: str):
        if not self.log_file_path:
            return
        try:
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(line.rstrip("\n") + "\n")
        except Exception:
            pass

    def _log_json(self, json_data: dict):
        if not self.log_file_path:
            return
        try:
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(json_data, ensure_ascii=False, indent=2) + "\n")
        except Exception:
            pass

    def _normalize_slot_list(self, value: object, *, limit: int) -> list[list]:
        if not isinstance(value, list):
            return []
        out: list[list] = []
        seen: set[tuple] = set()
        for item in value:
            day: object
            start: object
            end: object

            if isinstance(item, (list, tuple)) and len(item) >= 3:
                day, start, end = item[0], item[1], item[2]
            elif isinstance(item, str):
                m = re.search(
                    r"(?:\bDay\s*)?(\d+)\s*[:_,]?\s*([0-2]?\d:\d{2})\s*[-–—~]\s*([0-2]?\d:\d{2})",
                    item,
                    flags=re.IGNORECASE,
                )
                if not m:
                    continue
                day, start, end = m.group(1), m.group(2), m.group(3)
            elif isinstance(item, dict):
                day = item.get("day") if "day" in item else item.get("date")
                time_range = item.get("time_range") if isinstance(item.get("time_range"), str) else ""
                if time_range:
                    m = re.search(r"^\s*([0-2]?\d:\d{2})\s*[-–—~]\s*([0-2]?\d:\d{2})\s*$", time_range)
                    if not m:
                        continue
                    start, end = m.group(1), m.group(2)
                else:
                    start = item.get("start") if "start" in item else item.get("start_time")
                    end = item.get("end") if "end" in item else item.get("end_time")
                if day is None or start is None or end is None:
                    continue
            else:
                continue

            day_i = None
            if isinstance(day, bool):
                day_i = None
            elif isinstance(day, int):
                day_i = day
            elif isinstance(day, float):
                try:
                    day_i = int(day)
                except Exception:
                    day_i = None
            elif isinstance(day, str):
                m = re.search(r"(\d+)", day)
                if m:
                    try:
                        day_i = int(m.group(1))
                    except Exception:
                        day_i = None
            if day_i is None:
                continue
            if not isinstance(start, str) or not isinstance(end, str):
                continue
            start_s = start.strip()
            end_s = end.strip()
            if not re.match(r"^[0-2]?\d:\d{2}$", start_s):
                continue
            if not re.match(r"^[0-2]?\d:\d{2}$", end_s):
                continue
            start_min = self._hhmm_to_minutes(start_s)
            end_min = self._hhmm_to_minutes(end_s)
            if start_min is None or end_min is None or end_min <= start_min:
                continue
            key = (day_i, start_s, end_s)
            if key in seen:
                continue
            seen.add(key)
            out.append([day_i, start_s, end_s])
            if len(out) >= limit:
                break
        return out

    def _memory_covers_slot(self, agent_memory: dict, *, day: int, start_time: str, end_time: str) -> bool:
        if not isinstance(agent_memory, dict):
            return False
        summary = agent_memory.get("summary_data")
        if not isinstance(summary, dict):
            return False
        intervals = summary.get(str(day))
        if not isinstance(intervals, list):
            return False
        start_min = self._hhmm_to_minutes(start_time)
        end_min = self._hhmm_to_minutes(end_time)
        if start_min is None or end_min is None or end_min <= start_min:
            return False
        for item in intervals:
            if not isinstance(item, str) or "-" not in item:
                continue
            a, b = item.split("-", 1)
            a_min = self._hhmm_to_minutes(a.strip())
            b_min = self._hhmm_to_minutes(b.strip())
            if a_min is None or b_min is None:
                continue
            if a_min <= start_min and end_min <= b_min:
                return True
        return False

    def _verify_slot_with_model(self, slot_text: str, round_actions: dict[int, object], _candidate_slots: list[list] | None = None) -> bool:
        # Deterministic verifier for protocol runs (no extra LLM call).
        slot = LeaderAgent._parse_day_slot(slot_text or "")
        if not slot:
            return False
        key = (int(slot["day"]), slot["start_time"], slot["end_time"])

        for _aid, payload in sorted(round_actions.items()):
            if not isinstance(payload, dict):
                return False
            accepted = payload.get("a") if "a" in payload else payload.get("accept")
            rejected = payload.get("r") if "r" in payload else payload.get("reject")
            accepted_slots = self._normalize_slot_list(accepted, limit=200) if isinstance(accepted, list) else []
            rejected_slots = self._normalize_slot_list(rejected, limit=200) if isinstance(rejected, list) else []
            if key in {tuple(s) for s in rejected_slots}:
                return False
            if key not in {tuple(s) for s in accepted_slots}:
                return False

        return True

    def _protocol_instruction_discuss(self, duration_minutes: int) -> str:
        return (
            "PHASE: BROAD AVAILABILITY SCAN (Phase 0).\n"
            "Action: Identify your Top 2-4 free windows from these STANDARD BLOCKS:\n"
            "1. Morning: 08:00-12:00\n"
            "2. Afternoon: 14:00-18:00\n"
            "3. Evening: 20:00-24:00\n"
            f"Constraint: Select a block ONLY if you have at least {duration_minutes} minutes of CONTINUOUS free time inside it.\n"
            "Output JSON ONLY: {'broad': [[day, 'time'], ...]}\n"
            "CRITICAL: Use the EXACT block start/end times (e.g., 'Morning', 'Afternoon', 'Evening') in your JSON, NOT your specific free times."
        )

    def _memory_instruction(self, duration_minutes: int) -> str:
        """
        Generates the initial instruction for Round 1 (Style: FREE + MEMORY).
        """
        return (
            f"Meeting negotiation started. The Team Memory is currently empty.\n"
            f"Constraint: Slots must be exactly {duration_minutes} minutes long.\n"
            f"**Action Protocol:**\n"
            f"1. **Respond First:** If previous speakers proposed slots, you MUST vote (Yes/No) on them.\n"
            f"2. **Propose New:** Then, propose new candidate slots from your own schedule."
        )

    def run_task(self, *, duration_minutes: int, max_rounds: int, conversion: int, mode: str):
        global TOKEN_STATS, token_cost, model_name
        TOKEN_STATS = {
            "total_prompt": 0,
            "total_completion": 0,
            "total_completion_reasoning": 0,
            "total_completion_visible": 0,
            "total_total": 0,
            "by_phase": {},
        }
        token_cost = 0.0

        self._initialize_memories(duration_minutes, mode)
        task_prompt = self.leader.initiate_task(self.member_names, duration_minutes, max_rounds, self.days)
        print(f"**{self.leader.name}**: {task_prompt}\n")
        self._init_logger(task_prompt)

        # Leader instruction is only used for protocol+memory (passed into agents next round).
        if self.style == "protocol":
            self.last_leader_instruction = self._protocol_instruction_discuss(duration_minutes=duration_minutes)
        elif self.use_memory:
            self.last_leader_instruction = self._memory_instruction(duration_minutes=duration_minutes)
        else:
            self.last_leader_instruction = ""

        past_round_buffers: list[list[str]] = []
        completed_flag = False
        final_conclusion = ""
        rounds_used = max_rounds

        for round_num in range(1, max_rounds + 1):
            # Conversation window control:
            # - Only applies to pure free (style=free, use_memory=false).
            # - All memory-enabled runs rely on team_memory/leader and do not carry chat history.
            round_messages: list[str] = []
            if  self.style == "free" and not self.use_memory:
                self.conversation_history = []
                if conversion > 0 and past_round_buffers:
                    for buf in past_round_buffers[-conversion:]:
                        self.conversation_history.extend(buf)
            else:
                self.conversation_history = []

            print(f"--- Round {round_num} ---\n")
            self._log(f"--- Round {round_num} ---\n")

            leader_instruction_this_round = self.last_leader_instruction
            if self.use_memory and leader_instruction_this_round:
                print(f"{self.leader.name}: {leader_instruction_this_round}\n")
                self._log(f"{self.leader.name}: {leader_instruction_this_round}\n")
                leader_msg = f"{self.leader.name}: {leader_instruction_this_round}\n"
                self.conversation_history.append(leader_msg)
                round_messages.append(leader_msg)

            agent_payloads: dict[int, object] = {}
            for agent in self.agents:
                statement = agent.speak(
                    task_prompt,
                    self.conversation_history,
                    self.team_memory,
                    leader_instruction_this_round,
                    style=self.style,
                    max_rounds=max_rounds,
                    current_round=round_num,
                    use_memory=self.use_memory,
                )
                agent_payloads[agent.id] = statement
                text = json.dumps(statement, ensure_ascii=False) if isinstance(statement, dict) else str(statement)
                print(f"**{agent.name}**: {text}\n")
                msg = f"{agent.name}: {text}\n"
                self.conversation_history.append(msg)
                round_messages.append(msg)
                self._log(msg)

            completed, conclusion, instruction = self.leader.check_completion(
                round_messages,
                task_prompt,
                self.team_memory,
                max_rounds=max_rounds,
                current_round=round_num,
                last_leader_instruction=self.last_leader_instruction,
            )

            self.last_leader_instruction = instruction if self.use_memory else ""


            # Always log leader assessment (helps diagnose false negatives/positives).
            if self.use_memory:
                if self.style == "protocol":
                    self._update_team_memory_protocol(
                        leader_instruction_this_round=leader_instruction_this_round,
                        next_instruction=self.last_leader_instruction,
                        task_prompt=task_prompt,
                    )
                else:
                    self._update_team_memory(task_prompt, round_messages, self.last_leader_instruction)

                self._log(f"{self.leader.name}: {conclusion}\n")
                self._log(f"{self.leader.name}: {instruction}\n")


            print(f"Assessment: {'Completed' if completed else 'In Progress'}")
            print(f"Summary: {conclusion}\n")
            print(f"Instruction: {self.last_leader_instruction}\n")
            self._log(f"Round {round_num} token cost: {token_cost}")
            self._log_json(self.team_memory)

            if self.style == "free" and not self.use_memory:
                past_round_buffers.append(round_messages)

            if completed:
                completed_flag = True
                final_conclusion = conclusion
                rounds_used = round_num
                self._write_meta_json(
                    task_prompt=task_prompt,
                    max_rounds=max_rounds,
                    duration_minutes=duration_minutes,
                    completed=completed_flag,
                    rounds_used=rounds_used,
                    final_conclusion=final_conclusion,
                )
                return

        final_conclusion = final_conclusion or "No common meeting slot was found."
        rounds_used = max_rounds
        self._write_meta_json(
            task_prompt=task_prompt,
            max_rounds=max_rounds,
            duration_minutes=duration_minutes,
            completed=completed_flag,
            rounds_used=rounds_used,
            final_conclusion=final_conclusion,
        )

    def _update_team_memory_protocol(
        self,
        leader_instruction_this_round: str,
        next_instruction: str,
        task_prompt: str
    ):
        """
        Updates team memory based on the strict protocol phase transition defined by the Leader.
        """
        team_json = json.dumps(self.team_memory, ensure_ascii=False)


        prompt = f"""
            You are the **Protocol State Machine**.
            **Goal:** Update the Team Memory JSON based on the Leader's latest command.

            ### 1. INPUTS
            **Current Memory:** {team_json}
            **Leader's Command:** "{next_instruction}"

            ### 2. TRANSITION LOGIC (Check triggers in STRICT ORDER)

            **CASE A: BROAD STRATEGY SET (Trigger: "FOCUS ON")**
            - **Scenario:** Leader finished Broad Scan and defined the search scope.
            - **Action:**
            1. Extract the time blocks from the command (e.g., "Day 1 Morning").
            2. **Map to Standard Times:**
                - "Morning" -> "08:00", "12:00"
                - "Afternoon" -> "14:00", "18:00"
                - "Evening" -> "20:00", "24:00"
            3. Save result to `broad_time_blocks`: `[[1, "08:00", "12:00"], ...]`.
            4. Set `phase` = "COLLECT".
            5. Ensure `candidate_slots` is empty `[]`.

            **CASE B: START VOTING (Trigger: "PHASE: VOTE")**
            - **Scenario:** Leader listed specific numbered slots for voting.
            - **Action:**
            1. Extract the slots (e.g., "0: Day 1 10:00...") into `candidate_slots`.
            2. Keep `rejected_slots` and `broad_time_blocks` unchanged.
            3. Set `phase` = "VOTE".

            **CASE C: VOTE FAILED / RESET (Trigger: "PHASE: COLLECT" AND NO "FOCUS ON")**
            - **Scenario:** Voting failed, need to propose new slots.
            - **Action:**
            1. **Archive:** Move ALL items from current `candidate_slots` into `rejected_slots`.
            2. **Deduplicate:** Ensure `rejected_slots` contains unique items.
            3. **Clear:** Set `candidate_slots` to `[]`.
            4. Keep `broad_time_blocks` unchanged.
            5. Set `phase` = "COLLECT".

            **CASE D: DONE (Trigger: "Finalized" or "Success")**
            - **Action:** Return memory exactly as is.
            - **Set:** `phase` = "FINALIZED".

            ### 3. OUTPUT FORMAT
            Return JSON ONLY.
            {{
            "candidate_slots": [[day, "HH:MM", "HH:MM"], ...],
            "rejected_slots": [[day, "HH:MM", "HH:MM"], ...],
            "broad_time_blocks": [[day, "HH:MM", "HH:MM"], ...],
            "phase": "VOTE" | "COLLECT" | "FINALIZED"
            }}
            """


        updated_memory = call_llm_json_stream(
            prompt,
            tag="update_memory_protocol",
            role="system",
            phase="update_memory",
            task_type="meeting",
            style="protocol",
            enable_thinking=True
        )


        if isinstance(updated_memory, dict):

            cands = self._normalize_slot_list(updated_memory.get("candidate_slots"), limit=20)
            rejs = self._normalize_slot_list(updated_memory.get("rejected_slots"), limit=500)
            brand_blocks = self._normalize_slot_list(updated_memory.get("broad_time_blocks"), limit=10)
            phase = str(updated_memory.get("phase") or "").upper()

            self.team_memory["candidate_slots"] = cands
            self.team_memory["rejected_slots"] = rejs
            self.team_memory["broad_time_blocks"] = brand_blocks
            if phase in {"VOTE", "COLLECT", "FINALIZED"}:
                self.team_memory["phase"] = phase

    def _update_team_memory(self, task_prompt: str, turn_log: list[str], leader_instruction: str):
        team_json = json.dumps(self.team_memory, ensure_ascii=False)
        log_str = "".join(turn_log or [])
        prompt = f"""
            You are the **Meeting Memory Scribe**.
            **Goal:** Categorize every time slot mentioned in the log into "Candidate" (Alive) or "Rejected" (Dead).

            ### 1. INPUTS
            **Task:** {task_prompt}
            **Leader's Summary:** "{leader_instruction}"
            **Conversation Log:**
            {log_str}
            **Previous Memory:** {team_json}

            ### 2. CLASSIFICATION LOGIC (Execute Step-by-Step)

            **STEP A: IDENTIFY REJECTIONS (The "Kill" List)**
            - Scan the log for keywords: "No", "Busy", "Conflict", "Cannot do".
            - If ANY agent rejects a slot (e.g., "Day 1 10:00"), mark it as **REJECTED**.
            - *Global Kill:* If Leader says "Abandon Day 1", ALL Day 1 slots are **REJECTED**.

            **STEP B: IDENTIFY CANDIDATES (The "Live" List)**
            - Scan for slots currently being proposed or verified.
            - *Condition:* It must be proposed AND **NOT** rejected by anyone yet.
            - *Leader's Weight:* If Leader says "Verify Day 2 14:00", it is automatically a **CANDIDATE**.

            **STEP C: MERGE & CLEAN**
            - Add new Rejections to the `rejected_slots`.
            - Remove any Slot from `candidate_slots` if it appears in `rejected_slots`.
            - *Rule:* A slot cannot be both. Rejected wins.

            ### 3. OUTPUT FORMAT
            Return JSON ONLY.
            {{
            "candidate_slots": [[day, "HH:MM", "HH:MM"], ...],
            "rejected_slots": [[day, "HH:MM", "HH:MM"], ...]
            }}
            """
        updated = call_llm_json_stream(
            prompt,
            tag="update_memory",
            role="system",
            phase="update_memory",
            task_type="meeting",
            style="free",
            enable_thinking=True
        )
        if not isinstance(updated, dict) or not updated:
            return
        cand = self._normalize_slot_list(updated.get("candidate_slots"), limit=25)
        rej = self._normalize_slot_list(updated.get("rejected_slots"), limit=300)
        rej_set = {tuple(x) for x in rej}
        cand = [x for x in cand if tuple(x) not in rej_set]
        self.team_memory["candidate_slots"] = cand
        self.team_memory["rejected_slots"] = rej

    def _write_meta_json(
        self,
        *,
        task_prompt: str,
        max_rounds: int,
        duration_minutes: int,
        completed: bool,
        rounds_used: int,
        final_conclusion: str,
    ):
        if not self.meta_file_path:
            return
        evaluation = self._evaluate_against_ground_truth(duration_minutes=duration_minutes, final_conclusion=final_conclusion)
        meta = {
            "task_type": "meeting",
            "style": self.style,
            "use_memory": self.use_memory,
            "model": model_name,
            "thinking_override": thinking_override,
            "max_output_tokens": max_output_tokens,
            "schedule_dir": self.schedule_dir,
            "log_file": self.log_file_path,
            "start_time": self.run_start_time,
            "max_rounds": max_rounds,
            "duration_minutes": duration_minutes,
            "completed": completed,
            "rounds_used": rounds_used,
            "final_conclusion": final_conclusion,
            "members": self.member_names,
            "token_usage": {
                "total": {
                    "prompt": TOKEN_STATS["total_prompt"],
                    "completion": TOKEN_STATS["total_completion"],
                    "completion_reasoning": TOKEN_STATS["total_completion_reasoning"],
                    "completion_visible": TOKEN_STATS["total_completion_visible"],
                    "total": TOKEN_STATS["total_total"],
                },
                "by_phase": TOKEN_STATS["by_phase"],
            },
            "task_prompt": task_prompt,
            "command": " ".join(sys.argv) if sys.argv else "",
            "evaluation": evaluation,
        }
        try:
            with open(self.meta_file_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _extract_final_slot_via_llm(self, *, final_conclusion: str, duration_minutes: int) -> tuple[dict | None, dict]:
        if not isinstance(final_conclusion, str) or not final_conclusion.strip():
            return None, {"found": False}

        prompt = f"""
            You are a **Precision Regex Parsing Agent**.
            Your sole job is to extract the Final Meeting Slot from a summary string.

            ### INPUT TEXT
            "{final_conclusion}"

            ### EXTRACTION LOGIC (STRICT)
            1. **Target Pattern:** Look **ONLY** for the specific pattern: `Day [N] [HH:MM]-[HH:MM]`.
            2. **Anchor Rule:** The `day` value MUST be the integer immediately following the word "Day".
               - *Example:* In "Day 4", the day is 4.
               - *Counter-Example:* In "(5/5 votes)", the 5 is **NOT** the day. **IGNORE vote counts, indices, and scores.**
            3. **Time Parsing:** - Extract the first `HH:MM` as `start_time`.
               - Extract the second `HH:MM` as `end_time`.
               - If only `start_time` is found (e.g., "Day 1 14:00"), calculate: `end_time = start_time + {duration_minutes} minutes`.
            4. **Validation:**
               - If the text contains "Day 4", but you see "5/5 votes", output `day: 4`, NOT 5.
               - If multiple conflicting slots are found, return `"found": false`.

            ### OUTPUT FORMAT
            Return JSON ONLY.
            {{
                "found": boolean,
                "day": integer|null,
                "start_time": "HH:MM"|null,
                "end_time": "HH:MM"|null
            }}
            """
        raw = call_llm_json(
            prompt,
            tag="eval_extract",
            role="system",
            phase="evaluation_extract",
            task_type="meeting",
            style=self.style,
        )
        if not isinstance(raw, dict) or not raw:
            return None, {"found": False}

        if not bool(raw.get("found")):
            return None, raw

        day = raw.get("day")
        start_time = raw.get("start_time")
        end_time = raw.get("end_time")
        try:
            day_i = int(day)
        except Exception:
            return None, raw
        if day_i < 1:
            return None, raw
        if not isinstance(start_time, str) or not start_time.strip():
            return None, raw
        start_s = start_time.strip()
        end_s = str(end_time).strip() if isinstance(end_time, str) else ""

        start_min = self._hhmm_to_minutes(start_s)
        end_min = self._hhmm_to_minutes(end_s) if end_s else None
        if start_min is None:
            return None, raw
        expected_end_min = start_min + int(duration_minutes)
        if expected_end_min > 24 * 60:
            return None, raw
        if end_min is None or end_min <= start_min or (end_min - start_min) != int(duration_minutes):
            end_min = expected_end_min
            end_s = self._minutes_to_hhmm(end_min)

        return {"day": day_i, "start_time": start_s, "end_time": end_s}, raw

    def _evaluate_against_ground_truth(self, *, duration_minutes: int, final_conclusion: str) -> dict:
        gt_path = None
        for name in ("task_plans", "task_plans.json"):
            candidate = os.path.join(self.schedule_dir, name)
            if os.path.exists(candidate):
                gt_path = candidate
                break
        if not gt_path:
            return {"status": "no_ground_truth", "detail": "task_plans not found"}

        try:
            with open(gt_path, "r", encoding="utf-8") as f:
                gt_data = json.load(f)
        except Exception as e:
            return {"status": "gt_load_error", "error": str(e)}

        slot, extraction = self._extract_final_slot_via_llm(final_conclusion=final_conclusion, duration_minutes=duration_minutes)
        slot_source = "llm" if slot else "regex"
        if not slot:
            slot = LeaderAgent._parse_day_slot(final_conclusion or "")
        if not slot:
            return {
                "status": "no_slot",
                "normalized_result": {"task_type": "meeting", "slots": []},
                "ground_truth_file": gt_path,
                "slot_source": slot_source,
                "slot_extraction": extraction if isinstance(extraction, dict) else {},
            }

        time_unit = self.schedules[0].get("time_unit", "hours") if self.schedules else "hours"
        minutes_per_unit = 60
        if time_unit == "hours":
            minutes_per_unit = 60
        elif isinstance(time_unit, str) and time_unit.startswith("minutes_"):
            try:
                minutes_per_unit = int(time_unit.split("_")[1])
            except Exception:
                minutes_per_unit = 15
        units_per_hour = 60 // minutes_per_unit
        units_per_day = units_per_hour * 24

        start_min = self._hhmm_to_minutes(slot["start_time"]) or 0
        end_min = self._hhmm_to_minutes(slot["end_time"]) or (start_min + duration_minutes)
        if end_min <= start_min:
            end_min = start_min + minutes_per_unit
        duration_units = max(1, (end_min - start_min) // minutes_per_unit)
        start_local_units = start_min // minutes_per_unit
        day = slot["day"] if isinstance(slot.get("day"), int) and slot["day"] >= 1 else 1
        start_units = (day - 1) * units_per_day + start_local_units
        end_units = start_units + duration_units

        gt_entry = gt_data.get("meeting") or {}
        gt_solutions = gt_entry.get("solutions") or []
        matched_ids = [sol.get("solution_id") for sol in gt_solutions if sol.get("start_time") == start_units and sol.get("end_time") == end_units]

        return {
            "status": "ok" if matched_ids else "mismatch",
            "normalized_result": {"task_type": "meeting", "slots": [slot]},
            "ground_truth_file": gt_path,
            "slot_source": slot_source,
            "slot_extraction": extraction if isinstance(extraction, dict) else {},
            "time_unit": time_unit,
            "minutes_per_unit": minutes_per_unit,
            "chosen_slot_units": {"start": start_units, "end": end_units},
            "chosen_roster_by_day": None,
            "matched_solution_ids": matched_ids,
        }


def run_meeting(*, scenario_dir: str, t: int, max_rounds: int, conversion: int, style: str, use_memory: bool, mode: str):
    scenario_dir = os.path.abspath(scenario_dir)
    if not os.path.isdir(scenario_dir):
        raise FileNotFoundError(f"Scenario directory not found: {scenario_dir}")
    orchestrator = Orchestrator(scenario_dir, style=style, use_memory=use_memory)
    orchestrator.run_task(duration_minutes=t, max_rounds=max_rounds, conversion=conversion, mode=mode)


def main():
    parser = argparse.ArgumentParser(description="Meeting-only runner (Qwen-compatible endpoint)")
    parser.add_argument("--task-type", choices=["meeting"], default="meeting", help="Task type (fixed: meeting)")
    parser.add_argument("--scenario-dir", help="Directory containing Member*_final_schedule.json and task_plans", default="meeting/minutes_10/5days/4/1")
    parser.add_argument("--t", default=20, type=int, help="Meeting duration minutes (default: 20)")
    parser.add_argument("--max-rounds", type=int, default=5, help="Maximum negotiation rounds (default: 10)")
    parser.add_argument("--conversion", type=int, default=1, help="Conversation window control (only free+nomem). Default: 1")
    parser.add_argument("--protocol", type=str, choices=["true", "false"], default="true", help="Use protocol style (true/false)")
    parser.add_argument("--memory", type=str, choices=["true", "false"], default="true", help="Use memory (true/false)")
    parser.add_argument("--mode", type=str, choices=["serial", "parallel"], default="parallel", help="Initialize memory mode(serial/parallel)")
    parser.add_argument(
        "--provider",
        type=str,
        default="qwen",
        help="LLM provider key in .env (e.g. api_yi | qwen | sunapi). Uses <PROVIDER>_OPENAI_API_KEY and <PROVIDER>_OPENAI_BASE.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen-plus-2025-09-11",
        help="LLM model to use.",
    )
    parser.add_argument(
        "--thinking", choices=["auto", "true", "false"], default="auto",
        help="Override thinking for Qwen-compatible calls; auto preserves experiment defaults.",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=None,
        help="Optional per-response token limit; unset preserves the provider default.",
    )
    args = parser.parse_args()
    if args.max_tokens is not None and args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")

    style = "protocol" if args.protocol == "true" else "free"
    use_memory = args.memory == "true"
    global model_name, client, base_url, api_key, thinking_override, max_output_tokens
    model_name = args.model
    thinking_override = None if args.thinking == "auto" else args.thinking == "true"
    max_output_tokens = args.max_tokens
    mode = args.mode
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    dotenv_vars = _load_dotenv_file(env_path)
    api_key, base_url = _get_provider_config(args.provider, dotenv_vars)
    if OpenAI is not None:
        if httpx is not None:
            http_client = httpx.Client(timeout=600.0, trust_env=False)
            client = OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)
        else:
            client = OpenAI(api_key=api_key, base_url=base_url)
    else:
        client = None

    run_meeting(
        scenario_dir=args.scenario_dir,
        t=args.t,
        max_rounds=args.max_rounds,
        conversion=args.conversion,
        style=style,
        use_memory=use_memory,
        mode=mode
    )

if __name__ == "__main__":
    main()
