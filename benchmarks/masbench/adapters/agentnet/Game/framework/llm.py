import asyncio
import os

from openai import AsyncOpenAI

# Prefer the new google-genai SDK; fall back to google-generativeai if unavailable.
try:
    from google import genai as genai_new  # New SDK (google-genai)
    HAS_GOOGLE_GENAI = True
except Exception:
    genai_new = None
    HAS_GOOGLE_GENAI = False

try:
    import google.generativeai as genai  # Legacy SDK (google-generativeai)
except Exception:
    genai = None


def _env_bool(name, default=False):
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in ('1', 'true', 'yes', 'y', 'on')


def _env_float(name, default):
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _env_optional_float(name):
    raw = os.getenv(name)
    if raw is None:
        return None
    raw = str(raw).strip()
    if not raw:
        return None
    try:
        return float(raw)
    except Exception:
        return None


def _unsupported_sampling_param_error(exc):
    text = str(exc or '').lower()
    if not text:
        return False
    markers = (
        'unsupported',
        'unknown field',
        'unknown parameter',
        'unrecognized request argument',
        'extra_forbidden',
        'additional properties are not allowed',
        'not allowed',
        'invalid parameter',
    )
    return any(marker in text for marker in markers)


def _get_attr_or_key(obj, name, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _safe_int(value, default=0):
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _extract_openai_usage_counts(usage):
    prompt = _safe_int(_get_attr_or_key(usage, 'prompt_tokens', 0))
    completion_raw = _safe_int(_get_attr_or_key(usage, 'completion_tokens', 0))
    total = _safe_int(_get_attr_or_key(usage, 'total_tokens', prompt + completion_raw), prompt + completion_raw)
    details = _get_attr_or_key(usage, 'completion_tokens_details', None)
    reasoning = _safe_int(_get_attr_or_key(details, 'reasoning_tokens', 0))
    completion_visible = max(0, completion_raw - reasoning)
    return {
        'prompt': prompt,
        'completion_visible': completion_visible,
        'completion_raw': completion_raw,
        'reasoning': reasoning,
        'total': total,
    }


def _merge_sampling_kwargs(base_kwargs, sampling_override=None):
    merged = dict(base_kwargs or {})
    if not sampling_override:
        return merged
    override = dict(sampling_override or {})
    if override.pop('replace', False):
        merged = {}
    remove = override.pop('remove', ()) or ()
    if isinstance(remove, str):
        remove = (remove,)
    for key in remove:
        merged.pop(str(key), None)
    for key, value in override.items():
        if value is None:
            merged.pop(str(key), None)
        else:
            merged[str(key)] = value
    return merged


class Chat:
    def __init__(self, base_url, api_key, model='gpt-4o-mini', stream=False, memory=False, sampling_seed=None):
        self.model = model
        self.msg = []
        self.memory = memory
        self.sys_prompt = {'role': 'system', 'content': ''}
        self.stream = stream
        self.api_key = api_key
        self.base_url = base_url
        # Backward-compatible: usage = completion tokens total
        self.usage = 0
        # New detailed usage counters
        self.usage_prompt = 0
        self.usage_completion = 0
        self.usage_completion_raw = 0
        self.usage_reasoning = 0
        self.usage_total = 0
        # Message-phase only counters (filled by environment.message_phase)
        self.message_prompt_tokens = 0
        self.message_completion_tokens = 0
        self.sampling_seed = sampling_seed

    def _deterministic_request_kwargs(self):
        if not _env_bool('LLM_DETERMINISTIC', False):
            return {}
        kwargs = {
            'temperature': _env_float('LLM_TEMPERATURE', 0.0),
            'top_p': _env_float('LLM_TOP_P', 1.0),
        }
        if _env_bool('LLM_USE_SEED', True) and self.sampling_seed is not None:
            try:
                kwargs['seed'] = int(self.sampling_seed)
            except Exception:
                pass
        return kwargs

    def sampling_override_for(self, purpose=None):
        if str(purpose or '').strip().lower() != 'basic_routing':
            return None
        if _env_bool('LLM_BASIC_ROUTING_DISABLE_DETERMINISTIC', False):
            return {'replace': True}
        override = {}
        temperature = _env_optional_float('LLM_BASIC_ROUTING_TEMPERATURE')
        if temperature is not None:
            override['temperature'] = temperature
        top_p = _env_optional_float('LLM_BASIC_ROUTING_TOP_P')
        if top_p is not None:
            override['top_p'] = top_p
        if _env_bool('LLM_BASIC_ROUTING_DISABLE_SEED', False):
            override['remove'] = ['seed']
        return override or None

    async def _create_completion(self, client, request_kwargs):
        try:
            return await client.chat.completions.create(**request_kwargs)
        except Exception as exc:
            if not _unsupported_sampling_param_error(exc):
                raise
            relaxed = dict(request_kwargs)
            changed = False
            for key in ('seed', 'top_p', 'temperature'):
                if key in relaxed:
                    relaxed.pop(key, None)
                    changed = True
            if not changed:
                raise
            return await client.chat.completions.create(**relaxed)

    def _gen_wrapper(self, response):
        tmp = []
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                s = chunk.choices[0].delta.content
                tmp.append(s)
                yield s
        self.msg.append({'role': 'assistant', 'content': ''.join(tmp)})

    def _format_response(self, response):
        if self.stream:
            return self._gen_wrapper(response)
        else:
            s = response.choices[0].message.content
            self.msg.append({'role': 'assistant', 'content': s})
            return s

    def jump_back(self):
        while len(self.msg) > 0 and self.msg[-1]['role'] == 'assistant':
            self.msg = self.msg[:-1]
        if len(self.msg) > 0:
            self.msg = self.msg[:-1]

    def system(self, content):
        self.sys_prompt = {'role': 'system', 'content': content}

    async def generate(self, content, sampling_override=None):
        if not self.api_key:
            raise RuntimeError(f"Missing API key for model client: model={self.model}")
        self.msg.append({'role': 'user', 'content': content})
        client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

        request_kwargs = dict(
            model=self.model,
            messages=[self.sys_prompt] + (self.msg if self.memory else self.msg[-1:]),  # System prompt and message history
            stream=self.stream,
            timeout=3600,
        )
        request_kwargs.update(_merge_sampling_kwargs(self._deterministic_request_kwargs(), sampling_override))
        # Disable thinking by default for Flash models; allow an environment override.
        if _env_bool('MASBENCH_DISABLE_THINKING', False):
            request_kwargs['extra_body'] = {'thinking': {'type': 'disabled'}}
        response = await self._create_completion(client, request_kwargs)
        usage_counts = _extract_openai_usage_counts(getattr(response, 'usage', None))
        pt = usage_counts['prompt']
        ct = usage_counts['completion_visible']
        raw_ct = usage_counts['completion_raw']
        reasoning_tokens = usage_counts['reasoning']
        tt = usage_counts['total']
        # Accumulate detailed counters
        self.usage_prompt += pt
        self.usage_completion += ct
        self.usage_completion_raw += raw_ct
        self.usage_reasoning += reasoning_tokens
        self.usage_total += (tt if tt else (pt + ct))
        # Back-compat counter mirrors completion
        self.usage += ct
        await client.close()

        return self._format_response(response)


class GeminiChat:
    """Official Gemini API adapter with the same interface as Chat.
    - Calls official Gemini models through google-generativeai.
    - Supports .system(), .generate(), .jump_back(), .usage, .msg, and related attributes.
    - Async generate runs the synchronous SDK in a thread to avoid blocking.
    """
    def __init__(self, api_key, model='gemini-1.5-flash', stream=False, memory=False, sampling_seed=None):
        self.model = model
        self.msg = []
        self.memory = memory
        self.sys_prompt = {'role': 'system', 'content': ''}
        self.stream = stream  # Streaming is currently unused.
        self.api_key = api_key
        # Backward-compatible completion tokens
        self.usage = 0
        # Detailed counters
        self.usage_prompt = 0
        self.usage_completion = 0
        self.usage_completion_raw = 0
        self.usage_reasoning = 0
        self.usage_total = 0
        self.message_prompt_tokens = 0
        self.message_completion_tokens = 0
        self.sampling_seed = sampling_seed

        # Prefer the new SDK; otherwise use the legacy SDK.
        self._use_new_sdk = bool(HAS_GOOGLE_GENAI)
        if self._use_new_sdk:
            try:
                # The new SDK reads GEMINI_API_KEY/GOOGLE_API_KEY from the environment by default.
                self.client = genai_new.Client()
            except Exception:
                self._use_new_sdk = False
        if not self._use_new_sdk:
            if genai is None:
                raise ImportError("google-generativeai is not installed; run: pip install google-generativeai")
            try:
                genai.configure(api_key=self.api_key)
            except Exception:
                pass

    def jump_back(self):
        while len(self.msg) > 0 and self.msg[-1]['role'] == 'assistant':
            self.msg = self.msg[:-1]
        if len(self.msg) > 0:
            self.msg = self.msg[:-1]

    def system(self, content):
        self.sys_prompt = {'role': 'system', 'content': content or ''}

    def sampling_override_for(self, purpose=None):
        if str(purpose or '').strip().lower() != 'basic_routing':
            return None
        if _env_bool('LLM_BASIC_ROUTING_DISABLE_DETERMINISTIC', False):
            return {'replace': True}
        override = {}
        temperature = _env_optional_float('LLM_BASIC_ROUTING_TEMPERATURE')
        if temperature is not None:
            override['temperature'] = temperature
        top_p = _env_optional_float('LLM_BASIC_ROUTING_TOP_P')
        if top_p is not None:
            override['top_p'] = top_p
        if _env_bool('LLM_BASIC_ROUTING_DISABLE_SEED', False):
            override['remove'] = ['seed']
        return override or None

    def _to_gemini_contents(self, extra_user_text=None):
        """Convert internal message history to a Gemini contents list."""
        contents = []
        if self.memory:
            for m in self.msg:
                role = m.get('role')
                text = str(m.get('content') or '')
                if role == 'user':
                    contents.append({'role': 'user', 'parts': [text]})
                elif role == 'assistant':
                    contents.append({'role': 'model', 'parts': [text]})
        else:
            if extra_user_text is not None:
                contents.append({'role': 'user', 'parts': [str(extra_user_text)]})
        return contents

    async def generate(self, content, sampling_override=None):
        del sampling_override
        # Maintain consistent conversation history, matching the OpenAI adapter.
        self.msg.append({'role': 'user', 'content': content})

        if self._use_new_sdk:
            # Run the new SDK's synchronous call in the thread pool.
            def _call_new():
                # Combine system prompt and conversation history.
                hist_lines = []
                sys_text = self.sys_prompt.get('content', '')
                if sys_text:
                    hist_lines.append(sys_text)
                if self.memory:
                    for m in self.msg:
                        role = m.get('role')
                        text = str(m.get('content') or '')
                        if role == 'user':
                            hist_lines.append(f"User: {text}")
                        elif role == 'assistant':
                            hist_lines.append(f"Assistant: {text}")
                else:
                    hist_lines.append(str(content))
                contents = "\n".join(hist_lines)
                return self.client.models.generate_content(model=self.model, contents=contents)

            resp = await asyncio.to_thread(_call_new)
            text = getattr(resp, 'text', '') or ''
            # Usage statistics, if available.
            usage_meta = getattr(resp, 'usage_metadata', None)
            try:
                if usage_meta is not None:
                    pt = getattr(usage_meta, 'prompt_token_count', None)
                    ct = getattr(usage_meta, 'candidates_token_count', None)
                    tt = getattr(usage_meta, 'total_token_count', None)
                    pt_i = int(pt) if pt is not None else 0
                    ct_i = int(ct) if ct is not None else (int(tt) - pt_i if tt is not None else 0)
                    tt_i = int(tt) if tt is not None else (pt_i + ct_i)
                    self.usage_prompt += pt_i
                    self.usage_completion += ct_i
                    self.usage_total += tt_i
                    self.usage += ct_i
            except Exception:
                pass
        else:
            # Legacy SDK GenerativeModel.
            def _call_old():
                model = genai.GenerativeModel(
                    self.model,
                    system_instruction=self.sys_prompt.get('content', '')
                )
                contents = self._to_gemini_contents(extra_user_text=content)
                resp = model.generate_content(contents, request_options={"timeout": 3600})
                return resp

            resp = await asyncio.to_thread(_call_old)
            text = getattr(resp, 'text', None)
            if text is None:
                try:
                    cand = getattr(resp, 'candidates', []) or []
                    text = "\n".join([
                        "".join([p.text for p in (getattr(c, 'content', None) or {}).parts])
                        for c in cand
                        if getattr(getattr(c, 'content', None), 'parts', None)
                    ])
                except Exception:
                    text = ''
            usage_meta = getattr(resp, 'usage_metadata', None)
            try:
                if usage_meta is not None:
                    pt = getattr(usage_meta, 'prompt_token_count', None)
                    ct = getattr(usage_meta, 'candidates_token_count', None)
                    tt = getattr(usage_meta, 'total_token_count', None)
                    pt_i = int(pt) if pt is not None else 0
                    ct_i = int(ct) if ct is not None else (int(tt) - pt_i if tt is not None else 0)
                    tt_i = int(tt) if tt is not None else (pt_i + ct_i)
                    self.usage_prompt += pt_i
                    self.usage_completion += ct_i
                    self.usage_total += tt_i
                    self.usage += ct_i
            except Exception:
                pass

        # Append to history.
        self.msg.append({'role': 'assistant', 'content': text})
        return text
class UserAgent:
    def __init__(self, base_url, api_key, model='gpt-4o-mini', stream=False, memory=False):
        self.system_prompt = ''
        self.usage = 0

    async def generate(self, content, sampling_override=None):
        del sampling_override
        print('SYSTEM')
        print(self.system_prompt)
        print('PROMPT')
        print(content)
        return input('>>>')

    def system(self, content):
        self.system_prompt = content
