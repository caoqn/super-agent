"""Summarize recorded human interventions without imputing missing values."""


def summarize(rows):
    values = [row['human_interventions'] for row in rows if row.get('human_interventions') is not None]
    return {'measured_episodes': len(values), 'episodes_without_intervention': sum(value == 0 for value in values)}
