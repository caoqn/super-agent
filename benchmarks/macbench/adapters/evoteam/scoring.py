"""Delegate scoring to the installed native MAC-Bench implementation."""
def evaluation_fields(*args, **kwargs):
    from mac_bench.evaluation.reports.metrics import evaluation_fields as native
    return native(*args, **kwargs)
