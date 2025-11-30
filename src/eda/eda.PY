import os
import io
import boto3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# -------- CONFIG ----------
BUCKET = "nyc-taxi-ml-ops"
KEY = "processed/train_200k.csv"     # 200k cleaned data file
OUTPUT_DIR = "src/eda/outputs"       # save inside EDA folder
# ----------------------------


def save_plot(fig, name):
    """Save plot to local outputs folder."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(f"{OUTPUT_DIR}/{name}.png", bbox_inches="tight")
    plt.close(fig)


def main():

    print("Downloading 200k processed dataset from S3…")
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=BUCKET, Key=KEY)
    df = pd.read_csv(obj["Body"])
    print("Data loaded:", df.shape)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ==========================
    # BASIC SUMMARY & MISSING
    # ==========================
    summary = df.describe(include="all").transpose()
    missing = df.isnull().sum()

    summary.to_csv(f"{OUTPUT_DIR}/summary.csv")
    missing.to_csv(f"{OUTPUT_DIR}/missing_values.csv")

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # ==========================
    # DISTRIBUTIONS
    # ==========================
    for col in numeric_cols:
        fig = plt.figure(figsize=(8, 5))
        sns.histplot(df[col], kde=True, bins=50)
        plt.title(f"Distribution of {col}")
        save_plot(fig, f"dist_{col}")

    # ==========================
    # CORRELATION HEATMAP
    # ==========================
    corr = df[numeric_cols].corr()
    fig = plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    save_plot(fig, "correlation_heatmap")

    # ==========================
    # OUTLIERS BOXPLOTS
    # ==========================
    for col in numeric_cols:
        fig = plt.figure(figsize=(8, 5))
        sns.boxplot(x=df[col])
        plt.title(f"Outliers for {col}")
        save_plot(fig, f"boxplot_{col}")

    # ==========================
    # FARE VS DISTANCE
    # ==========================
    if {"fare_amount", "distance_km"}.issubset(df.columns):
        fig = plt.figure(figsize=(8, 5))
        sns.scatterplot(x=df["distance_km"], y=df["fare_amount"], alpha=0.3)
        plt.title("Fare vs Distance")
        save_plot(fig, "fare_vs_distance")

    # ==========================
    # HOURLY TRIP DISTRIBUTION
    # ==========================
    if "hour" in df.columns:
        fig = plt.figure(figsize=(8, 5))
        sns.countplot(x=df["hour"])
        plt.title("Trips by Hour")
        save_plot(fig, "trips_by_hour")

    if "weekday" in df.columns:
        fig = plt.figure(figsize=(8, 5))
        sns.countplot(x=df["weekday"])
        plt.title("Trips by Weekday")
        save_plot(fig, "trips_by_weekday")

    # ==========================
    # GEO SCATTER PLOTS
    # ==========================
    if {"pickup_latitude", "pickup_longitude"}.issubset(df.columns):
        fig = plt.figure(figsize=(6, 6))
        plt.scatter(df["pickup_longitude"],
                    df["pickup_latitude"], s=1, alpha=0.3)
        plt.title("Pickup Locations")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        save_plot(fig, "pickup_scatter")

    if {"dropoff_latitude", "dropoff_longitude"}.issubset(df.columns):
        fig = plt.figure(figsize=(6, 6))
        plt.scatter(df["dropoff_longitude"],
                    df["dropoff_latitude"], s=1, alpha=0.3)
        plt.title("Dropoff Locations")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        save_plot(fig, "dropoff_scatter")

    print("=== EDA on 200k dataset complete → saved inside src/eda/outputs ===")


if __name__ == "__main__":
    main()
