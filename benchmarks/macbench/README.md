# MAC-Bench

Install and download data from the [official MAC-Bench repository](https://github.com/rzchen777-wq/MAC_Bench). The AgentNet adapter is in `adapters/agentnet/`. Keep benchmark data and runtime images outside Git.


## Super Agent adapter

MAC-Bench uses the official task environment and evaluator. Its primary metrics are Progress Score (PS) and Resolution Rate (RR). This directory does not include task data or the official evaluator; use the separate [MAC-Bench repository](https://github.com/rzchen777-wq/MAC_Bench) and pin its code and data versions.

`benchmarks/macbench/adapters/super_agent/adapter.py` invokes the official MAC-Bench `superagent` baseline and evaluator. Its configuration is in `methods/super_agent/config/macbench.yaml` and is passed to the evaluator by default. It records model connectivity, the shared backbone, the shared episode world, and the per-generation limit. Set task budgets through command-line options.

The official setup requires Linux, Python 3.11+, Node.js 24+, pnpm, and either Modal or Daytona. Follow the upstream documentation to install `.[modal,agents,superagent]` or the corresponding Daytona dependencies, then install pnpm dependencies in the official Superagent directory. Configure model and environment credentials and download the data. Example:

```sh
python benchmarks/macbench/adapters/super_agent/adapter.py \
  --upstream /path/to/MAC_Bench --python /path/to/MAC_Bench/.venv/bin/python \
  --task-id TASK_ID --output-dir results/macbench-dev \
  --max-seconds 7200 --method-config methods/super_agent/config/macbench.yaml --dry-run
```

Remove `--dry-run` to run the evaluation. The upstream checkout, environment, data, and credentials are not configured here, and this entry point has not completed end-to-end validation. Treat the official output as the source of final PS/RR metrics, usage, and traces.

The official MAC-Bench `superagent` implementation includes its task environment, router, workers, subagents, and evaluator. This adapter pins the upstream checkout, passes the YAML configuration and budgets, and saves the official output.

## Manager + 3 workers

The repository also includes a prompt-driven `manager3` implementation in
`methods/manager3/macbench/`. It is a fixed hierarchy of one manager and three
persistent workers. The first manager action must assign all three workers;
worker model calls are concurrent, while environment mutations are serialized
against the shared MAC-Bench episode world. The official MAC-Bench budget table
is applied per task and shared by all four agents.

Use `benchmarks/macbench/adapters/manager3/adapter.py` as the benchmark-facing
entry point, and `methods/manager3/config/macbench.yaml` as the method record and
`configs/manager3/macbench.env.example` as the configuration template. The
runner mounts the upstream MAC-Bench checkout at runtime instead of committing
its data or evaluator here. See `methods/manager3/README.md#mac-bench` for
local and Docker commands.
