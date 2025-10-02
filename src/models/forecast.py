from typing import Dict
import pandas as pd


def naive_monthly_forecast(history: pd.DataFrame, periods: int = 3) -> Dict:
    """
    Given a history DataFrame with columns ['year','month','total_amount'],
    return a naive forecast dict for next N months.
    """
    history = history.sort_values(['year', 'month']).reset_index(drop=True)
    history['key'] = history.apply(lambda r: f"{int(r['year'])}-{int(r['month']):02d}", axis=1)
    history_series = {str(k): float(v) for k, v in zip(history['key'], history['total_amount'])}

    if len(history) == 0:
        return {'history': {}, 'forecast': {}}

    last_year = int(history.iloc[-1]['year'])
    last_month = int(history.iloc[-1]['month'])
    last_value = float(history.iloc[-1]['total_amount'])

    forecast = {}
    year, month = last_year, last_month
    for _ in range(periods):
        month += 1
        if month > 12:
            month = 1
            year += 1
        key = f"{year}-{month:02d}"
        forecast[key] = last_value

    return {'history': history_series, 'forecast': forecast}


