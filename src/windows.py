
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def make_windows(df, history=24):
    """
    Create next-hour PM2.5 forecasting examples.

    X: previous `history` hourly PM2.5 values
    y: next-hour PM2.5 value

    Excludes missing values and time gaps.
    Never mixes stations.
    """

    if history < 1:
        raise ValueError("history must be positive")

    all_X = []
    all_y = []
    all_stations = []
    all_times = []

    for station, group in df.groupby("station"):

        group = group.sort_values("datetime")

        values = group["PM2.5"].to_numpy(dtype=float)
        times = group["datetime"].to_numpy(
            dtype="datetime64[ns]"
        )

        if len(group) <= history:
            continue

        # Each window includes historical values + target
        value_windows = sliding_window_view(
            values, window_shape=history + 1
        )

        # Ensure timestamps are consecutive hours
        hourly_steps = (
            np.diff(times) == np.timedelta64(1, "h")
        )

        consecutive = sliding_window_view(
            hourly_steps, window_shape=history
        ).all(axis=1)

        # Exclude missing or nonfinite measurements
        complete = np.isfinite(value_windows).all(axis=1)

        valid = consecutive & complete

        selected = value_windows[valid]

        if len(selected) == 0:
            continue

        all_X.append(selected[:, :-1])
        all_y.append(selected[:, -1])

        all_stations.extend(
            [station] * len(selected)
        )

        all_times.extend(
            times[history:][valid]
        )

    if not all_X:
        raise ValueError("No valid forecasting windows found")

    X = np.concatenate(all_X)
    y = np.concatenate(all_y)

    return (
        X,
        y,
        np.asarray(all_stations),
        np.asarray(all_times)
    )
