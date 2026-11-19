"""Task-local, shared admission accounting (including concurrent model requests)."""
from __future__ import annotations

import json
import threading
import time


class TaskBudgetExceeded(RuntimeError):
    pass


def prompt_token_reservation(messages, tools=None):
    # UTF-8 bytes are a conservative bound for byte-based provider tokenizers.
    # Include schemas and framing; settle against provider usage on completion.
    return len(json.dumps([messages, tools], ensure_ascii=False, default=str).encode('utf-8')) + 256


class TaskBudget:
    def __init__(self, limits):
        self.limits = dict(limits)
        for key in ('max_tokens', 'max_model_calls', 'max_tool_calls', 'max_env_actions', 'max_seconds'):
            if self.limits.get(key, 0) <= 0:
                raise ValueError(f'Missing or invalid task budget: {key}')
        self.started = time.monotonic()
        self.reserve_tokens = min(16384, int(self.limits['max_tokens']) // 4)
        self.submit_seconds = min(90.0, self.limits['max_seconds'] * .1)
        self.input_tokens = self.output_tokens = self.uncertain_tokens = 0
        self.model_calls = self.model_call_attempts = self.calls = 0
        self.reserved_tokens = 0
        self.reason = ''
        self.model_enabled = lambda: True
        self._lock = threading.RLock()
        self._ended = None

    def remaining_seconds(self):
        return max(0., self.limits['max_seconds'] - (time.monotonic() - self.started))

    def _stop(self, reason):
        self.reason = self.reason or reason
        raise TaskBudgetExceeded(self.reason)

    def submission_reason(self):
        with self._lock:
            if self.remaining_seconds() <= self.submit_seconds:
                self.reason = self.reason or 'wall_seconds'
            if self.model_call_attempts >= self.limits['max_model_calls'] - 1:
                self.reason = self.reason or 'model_calls'
            if self.calls >= min(self.limits['max_tool_calls'], self.limits['max_env_actions']):
                self.reason = self.reason or 'tool_calls'
            if self.input_tokens + self.output_tokens + self.uncertain_tokens >= self.limits['max_tokens'] - self.reserve_tokens:
                self.reason = self.reason or 'tokens'
            return self.reason

    def admit_model(self, prompt_tokens, output_tokens, emergency=False):
        with self._lock:
            if self.remaining_seconds() <= 0:
                self._stop('wall_seconds')
            if not emergency and self.submission_reason():
                self._stop(self.reason)
            call_limit = self.limits['max_model_calls'] - (0 if emergency else 1)
            if self.model_call_attempts >= call_limit:
                self._stop('model_calls')
            available = (self.limits['max_tokens'] - self.input_tokens - self.output_tokens
                         - self.uncertain_tokens - self.reserved_tokens
                         - (0 if emergency else self.reserve_tokens))
            output_tokens = min(output_tokens, available - prompt_tokens)
            if output_tokens <= 0:
                self._stop('tokens')
            reservation = prompt_tokens + output_tokens
            self.reserved_tokens += reservation
            self.model_call_attempts += 1
            return reservation, output_tokens

    def settle_model(self, reservation, response=None):
        with self._lock:
            self.reserved_tokens -= reservation
            usage = getattr(response, 'usage', None)
            if isinstance(usage, dict):
                prompt, output = usage.get('prompt_tokens'), usage.get('completion_tokens')
            else:
                prompt = getattr(usage, 'prompt_tokens', None)
                output = getattr(usage, 'completion_tokens', None)
            if type(prompt) is int and type(output) is int and min(prompt, output) >= 0:
                self.input_tokens += prompt
                self.output_tokens += output
            else:
                # A cancelled/failed request may still have consumed tokens upstream.
                self.uncertain_tokens += reservation
            if response is not None:
                self.model_calls += 1

    def admit(self):
        with self._lock:
            if self.submission_reason():
                self._stop(self.reason)
            self.calls += 1

    def finish(self):
        with self._lock:
            if self._ended is None:
                self._ended = time.monotonic()

    def snapshot(self):
        with self._lock:
            return {
                'limits': self.limits, 'input_tokens': self.input_tokens,
                'output_tokens': self.output_tokens,
                'total_tokens': self.input_tokens + self.output_tokens,
                'uncertain_tokens_reserved': self.uncertain_tokens,
                'inflight_tokens_reserved': self.reserved_tokens,
                'model_calls': self.model_calls, 'model_call_attempts': self.model_call_attempts,
                'tool_calls': self.calls, 'env_actions': self.calls,
                'emergency_token_reserve': self.reserve_tokens,
                'emergency_time_reserve_seconds': self.submit_seconds,
                'submission_reason': self.reason,
                'wall_seconds': round((self._ended or time.monotonic()) - self.started, 3),
                'token_limit_exceeded': self.input_tokens + self.output_tokens > self.limits['max_tokens'],
            }
