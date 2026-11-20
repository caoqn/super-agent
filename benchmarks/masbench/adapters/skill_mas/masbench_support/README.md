# Game worker

`game_worker.py` runs the separately installed MASBench Game environment in an isolated subprocess with JSON-lines RPC and no model calls. It exposes local observation, communication range, one pending action per actor per round, and native resource scoring. Physics advances only after every actor submits. Episodes end after at most 50 simultaneous rounds. Upstream physics and logging source are loaded from `MASBENCH_ROOT/Game`; install Python dependencies in the active environment. No private dependency directory or original machine path is required.
