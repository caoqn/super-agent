# MASBench

Obtain task inputs from the [official MASBench repository](https://github.com/BUPT-GAMMA/MASBench). The selected task-ID manifest is tracked in `data/manifests/`; generated inputs and run outputs remain local. Method-specific adapters are under `methods/<method>/masbench/`.

ReAct and vanilla currently share scoring helpers from the ReAct adapter. Preserve that scoring behavior when refactoring.
