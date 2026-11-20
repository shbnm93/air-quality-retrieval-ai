from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"


def load_air_quality_data(verbose=True):
    """Load and combine Beijing air-quality CSV files."""

    # Step 1: Find station files
    files = sorted(RAW_DIR.rglob("PRSA_Data_*.csv"))

    if not files:
        raise FileNotFoundError(
            "No station CSV files found in data/raw/"
        )

    if verbose:
        print(f"Found {len(files)} CSV files")

    # Step 2: Read each file
    dataframes = []

    for file in files:
        df = pd.read_csv(file)

        if verbose:
            print(f"Loading {file.name}: {len(df)} rows")

        dataframes.append(df)

    # Step 3: Combine all stations
    combined = pd.concat(
        dataframes,
        ignore_index=True
    )

    # Step 4: Build a timestamp
    combined["datetime"] = pd.to_datetime(
        combined[["year", "month", "day", "hour"]]
    )

    # Step 5: Sort by station and timestamp
    combined = combined.sort_values(
        ["station", "datetime"]
    ).reset_index(drop=True)

    # Step 6: Check for duplicate timestamps
    duplicates = combined.duplicated(
        subset=["station", "datetime"]
    ).sum()

    if duplicates:
        raise ValueError(
            f"Found {duplicates} duplicate station timestamps"
        )

    return combined


if __name__ == "__main__":

    data = load_air_quality_data()

    print("\nDataset shape:")
    print(data.shape)

    print("\nFirst five rows:")
    print(data.head())

    print("\nNumber of stations:")
    print(data["station"].nunique())

    print("\nMissing values:")
    print(data.isna().sum())
