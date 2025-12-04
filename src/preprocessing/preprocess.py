import boto3
import pandas as pd
import io
import math
from sklearn.model_selection import train_test_split

BUCKET = "nyc-taxi-ml-ops"
INPUT_KEY = "data/processed/train_200k.csv"

OUTPUT_TRAIN = "data/processed/train_final.csv"
OUTPUT_TEST = "data/processed/test_final.csv"


def haversine_distance(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (
        math.sin(dlat / 2)**2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    )
    c = 2 * math.asin(math.sqrt(a))
    return 6371 * c  # km


def add_features(df):
    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"], errors="coerce")
    df = df.dropna(subset=["pickup_datetime"])

    df["hour"] = df["pickup_datetime"].dt.hour
    df["month"] = df["pickup_datetime"].dt.month
    df["weekday"] = df["pickup_datetime"].dt.weekday

    df["distance_km"] = df.apply(
        lambda r: haversine_distance(
            r["pickup_latitude"], r["pickup_longitude"],
            r["dropoff_latitude"], r["dropoff_longitude"]
        ),
        axis=1
    )

    return df


def clean_data(df):
    df = df[
        (df["pickup_latitude"].between(40, 42)) &
        (df["dropoff_latitude"].between(40, 42)) &
        (df["pickup_longitude"].between(-75, -72)) &
        (df["dropoff_longitude"].between(-75, -72))
    ]

    df = df[
        (df["fare_amount"] > 0) &
        (df["fare_amount"] < 200) &
        (df["passenger_count"] > 0) &
        (df["passenger_count"] <= 6) &
        (df["distance_km"] > 0) &
        (df["distance_km"] < 100)
    ]

    return df


def main():
    s3 = boto3.client("s3")

    print("Downloading 200k dataset from S3...")
    obj = s3.get_object(Bucket=BUCKET, Key=INPUT_KEY)
    df = pd.read_csv(io.BytesIO(obj["Body"].read()))

    print("Starting preprocessing...")
    df = add_features(df)
    df = clean_data(df)

    print("Final cleaned shape:", df.shape)

    # Split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Upload train
    train_buf = io.StringIO()
    train_df.to_csv(train_buf, index=False)
    s3.put_object(Bucket=BUCKET, Key=OUTPUT_TRAIN,
                  Body=train_buf.getvalue().encode("utf-8"))

    # Upload test
    test_buf = io.StringIO()
    test_df.to_csv(test_buf, index=False)
    s3.put_object(Bucket=BUCKET, Key=OUTPUT_TEST,
                  Body=test_buf.getvalue().encode("utf-8"))

    print("Uploaded:")
    print(" →", f"s3://{BUCKET}/{OUTPUT_TRAIN}")
    print(" →", f"s3://{BUCKET}/{OUTPUT_TEST}")


if __name__ == "__main__":
    main()
