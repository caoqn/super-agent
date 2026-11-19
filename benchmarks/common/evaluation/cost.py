"""Summarize measured cost without imputing missing values."""


def summarize(rows):
    values = [row['cost_usd'] for row in rows if row.get('cost_usd') is not None]
    return {'measured_episodes': len(values), 'total_usd': sum(values) if values else None}
