from data_loader import load_air_quality_data


def inspect_data_quality(df):

    print("\n=== DATASET SUMMARY ===")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Stations:", df["station"].nunique())

    print("\n=== DATE RANGE ===")
    print("Start:", df["datetime"].min())
    print("End:", df["datetime"].max())

    print("\n=== MISSING VALUES (%) ===")
    missing_percentage = (
        df.isna().mean() * 100
    ).sort_values(ascending=False)

    print(missing_percentage.round(2))

    print("\n=== ROWS PER STATION ===")
    print(df.groupby("station").size())

    print("\n=== PM2.5 STATISTICS ===")
    print(df["PM2.5"].describe())


if __name__ == "__main__":
    data = load_air_quality_data()
    inspect_data_quality(data)