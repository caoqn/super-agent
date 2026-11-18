from framework.llm import Chat, UserAgent, GeminiChat


class ModelConfig:
    model: str
    api_base: str
    api_key: str
    provider: str = 'OPENAI'

    def as_dict(self):
        """Return metadata safe to serialize in experiment logs."""
        return {
            'model': self.model,
            'api_base': self.api_base,
            'api_key': '[REDACTED]',
            'provider': getattr(self, 'provider', 'OPENAI')
        }


class Framework:

    def __init__(self):
        self.env = None
        self.agents = None
        self.logger = None

    def start(self, agent_cls, env_cls, logger_cls, env_name, num_agents, model,
              agent_args=None, logger_args=None, env_args=None):
        if agent_args is None:
            agent_args = {}

        if logger_args is None:
            logger_args = {}

        if env_args is None:
            env_args = {}

        if isinstance(model, ModelConfig):
            model = [model for _ in range(num_agents)]
        self.agents = {}
        for i in range(num_agents):
            agent_name = f'Agent_{i}'
            prov = (getattr(model[i], 'provider', 'OPENAI') or 'OPENAI').upper()
            sampling_seed = None
            try:
                sampling_seed = int((env_args or {}).get('seed'))
            except Exception:
                sampling_seed = None
            if prov == 'GEMINI':
                brain = GeminiChat(model[i].api_key, model=model[i].model, sampling_seed=sampling_seed)
            else:
                brain = Chat(model[i].api_base, model[i].api_key, model[i].model, sampling_seed=sampling_seed)
            self.agents[agent_name] = agent_cls(agent_name, brain, **agent_args)

        self.logger = logger_cls(env_name, **logger_args)

        self.env = env_cls(env_name, self.agents, self.logger, **env_args)

        self.env.start()

        return self.env
