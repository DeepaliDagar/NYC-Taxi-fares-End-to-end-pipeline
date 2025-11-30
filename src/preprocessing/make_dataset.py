import boto3
import pandas as pd
import io

BUCKET = "nyc-taxi-ml-ops"
RAW_KEY = "raw/train.csv"
OUTPUT_KEY = "processed/train_200k.csv"

SAMPLE_SIZE = 200_000  # 200k rows


def main():
    s3 = boto3.client("s3")

    print("Downloading raw/train.csv from S3...")
    obj = s3.get_object(Bucket=BUCKET, Key=RAW_KEY)
    df = pd.read_csv(io.BytesIO(obj["Body"].read()))

    print("Raw shape:", df.shape)

    # Sample 200k rows
    if len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=42)

    print("Sampled shape:", df.shape)

    # Convert DataFrame to CSV bytes
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)

    # Upload back to S3
    print("Uploading processed 200k dataset to S3...")
    s3.put_object(
        Bucket=BUCKET,
        Key=OUTPUT_KEY,
        Body=csv_buffer.getvalue().encode("utf-8")
    )

    print("DONE! File saved to:", f"s3://{BUCKET}/{OUTPUT_KEY}")


if __name__ == "__main__":
    main()
