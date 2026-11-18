from traceback import print_exc
import os


class Agent:

    def __init__(self, name, brain, sys_prompt):
        self.name = name
        self.brain = brain
        self.msgs = {}
        self.sys_prompt = sys_prompt
        self.brain.system(self.sys_prompt)

    async def decision(self, obs):
        max_retries_raw = os.getenv('AGENT_DECISION_MAX_RETRIES', '2')
        try:
            max_retries = max(1, int(max_retries_raw))
        except Exception:
            max_retries = 2
        last_error = None
        for attempt in range(max_retries):
            try:
                prompt = self.gen_prompt(obs, self.msgs)
                text = await self.brain.generate(prompt)
                # Record action-phase response memory if supported
                try:
                    if hasattr(self, 'record_response'):
                        # Phase: 'action'
                        self.record_response(text, 'action', obs)
                except Exception:
                    pass
                res = self.to_action(text)
                self.msgs = {}
                return res
            except Exception as e:
                last_error = e
                print_exc()
                if attempt + 1 >= max_retries:
                    raise RuntimeError(
                        f"{self.name} decision failed after {max_retries} attempts"
                    ) from e
        self.msgs = {}
        raise RuntimeError(f"{self.name} decision failed") from last_error

    def notify(self, name, msg):
        self.msgs[name] = msg

    def gen_prompt(self, obs, msgs):
        return ''

    def to_action(self, response):
        return {}
