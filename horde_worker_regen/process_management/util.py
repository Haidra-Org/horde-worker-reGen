from datetime import datetime

_time_units = [
    ("year", 365 * 24 * 3600),
    ("month", 30 * 24 * 3600),
    ("day", 24 * 3600),
    ("hour", 3600),
    ("minute", 60),
]


def dt_to_td_str(dt: datetime) -> str | None:
    now = datetime.now()
    time_difference = (now - dt).total_seconds()

    chosen: tuple[str, int] | None = None

    for unit, seconds_in_unit in _time_units:
        if time_difference >= seconds_in_unit:
            chosen = (unit, seconds_in_unit)

    if chosen is None:
        chosen = ("second", 1)

    unit, seconds_in_unit = chosen
    count = int(time_difference / seconds_in_unit)
    return f"{count} {unit}{'' if count == 1 else 's'} ago"
