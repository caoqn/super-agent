"""Summarize measured wall-clock latency."""


def summarize(rows):
    values = [row['latency_seconds'] for row in rows if row.get('latency_seconds') is not None]
    return {'measured_episodes': len(values), 'mean_seconds': sum(values) / len(values) if values else None}
