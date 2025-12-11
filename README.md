# NYC Taxi Fare Prediction - End-to-End MLOps Pipeline

A production-ready machine learning system that predicts taxi fare amounts in New York City using AWS SageMaker, automated feature engineering, and comprehensive model monitoring.

## Challenge & Objective

**Challenge**: A simple distance-based estimate gives an RMSE of $5–$8

**Objective**: Use advanced feature engineering and ML models to significantly outperform this baseline by building a robust, monitored, and production-ready ML system.

## 📊 Dataset

**Source**: Kaggle - NYC Taxi Fare Prediction  
**Total Rows**: ~200k rows  
**Train/Test Split**: 80/20

### Key Features

- **pickup_datetime** – timestamp of the ride
- **pickup_longitude** & **pickup_latitude** – pickup location coordinates
- **dropoff_longitude** & **dropoff_latitude** – dropoff location coordinates
- **passenger_count** – number of passengers

### Target Variable

- **fare_amount** (USD) - total fare including tolls

## 🏗️ Architecture

This project implements a complete MLOps pipeline on AWS with the following components:

```
┌─────────────────┐
│   Data Source   │ (S3: Raw Data)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Preprocessing  │ (Feature Engineering + Cleaning)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  EDA & Analysis │ (Visualizations & Insights)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Model Training │ (SageMaker Autopilot/AutoML)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Registry  │ (SageMaker Model Registry)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Deployment    │ (SageMaker Real-time Endpoint)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Monitoring    │ (Evidently AI + SageMaker Model Monitor)
└─────────────────┘
```

## Project Structure

```
NYC-Taxi-fares-End-to-end-pipeline/
├── dags/                      # Airflow DAGs for orchestration
│   └── utils/                 # Airflow utility functions
├── data/
│   └── processed/             # Processed train/test datasets
├── models/
│   └── baseline.py            # Baseline model implementation
├── sagemaker/
│   ├── pipeline.py            # SageMaker pipeline orchestration
│   └── drift_check.py         # Drift detection with Evidently
├── src/
│   ├── automl/
│   │   ├── automl.py          # AutoML training logic
│   │   └── track_experiments.py  # MLflow experiment tracking
│   ├── eda/
│   │   ├── eda.py             # Exploratory Data Analysis
│   │   └── outputs/           # EDA visualizations and reports
│   ├── evaluation/
│   │   └── test_endpoint.py   # Endpoint evaluation scripts
│   ├── inference/
│   │   └── deployment.py      # Model deployment automation
│   ├── monitoring/            # Monitoring scripts
│   ├── pipeline/
│   │   └── pipeline.py        # Main pipeline orchestration
│   └── preprocessing/
│       ├── make_dataset.py    # Dataset creation utilities
│       └── preprocess.py      # Feature engineering & data cleaning
└── infra/                     # Infrastructure as Code
    ├── iam/                   # IAM roles and policies
    ├── mwaa/                  # Amazon Managed Workflows for Apache Airflow
    └── s3/                    # S3 bucket configurations
```

## Key Features

### 1. Advanced Feature Engineering

The preprocessing pipeline includes:

- **Temporal Features**: Extract hour, month, and weekday from pickup_datetime
- **Distance Calculation**: Haversine distance between pickup and dropoff coordinates
- **Data Cleaning**:
  - Geographic bounds filtering (NYC area: lat 40-42, lon -75 to -72)
  - Fare amount validation ($0-$200 range)
  - Passenger count validation (1-6 passengers)
  - Distance validation (0-100 km)

### 2. Automated Model Training

- **SageMaker Autopilot**: Automated model selection and hyperparameter tuning
- **MLflow Integration**: Experiment tracking with SageMaker Managed MLflow
- **Multiple Candidates**: Evaluates multiple model architectures automatically
- **Best Model Selection**: Automatically selects the best performing candidate

### 3. Model Management

- **SageMaker Model Registry**: Centralized model versioning and management
- **Model Approval Workflow**: Manual approval process for production deployment
- **Model Artifact Storage**: Versioned model artifacts stored in S3

### 4. Real-time Inference

- **SageMaker Real-time Endpoints**: Low-latency prediction endpoints
- **Automatic Scaling**: Handles variable inference loads
- **Batch Inference Support**: For large-scale predictions

### 5. Comprehensive Monitoring

#### Evidently AI Integration
- **Data Drift Detection**: Monitors input feature distributions
- **Model Performance Monitoring**: Tracks prediction quality over time
- **Interactive Reports**: HTML reports with visualizations
- **S3 Integration**: Reports stored in S3 for easy access

#### SageMaker Model Monitor
- **Baseline Statistics**: Establishes baseline from training data
- **Statistical Tests**: KS test for numeric features, Chi-square for categorical
- **Automatic Alerts**: Notifications when drift is detected

## Getting Started

### Prerequisites

- AWS Account with appropriate permissions
- Python 3.8+
- AWS CLI configured with credentials
- SageMaker execution role with necessary permissions

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd NYC-Taxi-fares-End-to-end-pipeline
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Additional dependencies** (install as needed):
```bash
pip install sagemaker boto3 mlflow evidently pandas scikit-learn matplotlib seaborn
```

### Configuration

1. **Update S3 bucket configuration** in the respective scripts:
   - `src/preprocessing/preprocess.py`: Update `BUCKET` variable
   - `models/baseline.py`: Update `BUCKET` and `MLFLOW_SERVER_NAME`
   - `sagemaker/drift_check.py`: Update `S3_BUCKET` and `ENDPOINT_NAME`

2. **Set up SageMaker Execution Role**:
   - Create or use an existing SageMaker execution role
   - Ensure it has permissions for S3, SageMaker, and MLflow

3. **Configure MLflow Tracking**:
   - Set up SageMaker Managed MLflow (or use local MLflow)
   - Update `MLFLOW_SERVER_NAME` in training scripts

## Usage

### 1. Data Preprocessing

Run the preprocessing pipeline to engineer features and clean the data:

```bash
python src/preprocessing/preprocess.py
```

This will:
- Load data from S3 or local storage
- Engineer temporal and distance features
- Clean invalid records
- Split into train/test sets (80/20)
- Upload processed data to S3

### 2. Exploratory Data Analysis

Generate visualizations and insights:

```bash
python src/eda/eda.py
```

Outputs include:
- Distribution plots for all numeric features
- Correlation heatmap
- Box plots for outlier detection
- Scatter plots for geographic patterns
- Summary statistics and missing value reports

### 3. Model Training

#### Option A: Using SageMaker Autopilot (Automated)

```bash
python models/baseline.py
```

This will:
- Start a SageMaker Autopilot job
- Automatically explore multiple model candidates
- Track experiments in MLflow
- Select the best performing model

#### Option B: Using Pipeline Script

```bash
python src/pipeline/pipeline.py
```

This orchestrates:
- AutoML training
- MLflow logging
- Model registration in SageMaker Model Registry

### 4. Model Deployment

Deploy the approved model from the registry:

```bash
export SAGEMAKER_MODEL_PACKAGE_ARN=<your-model-package-arn>
export AWS_REGION=us-east-2
python src/inference/deployment.py
```

### 5. Model Monitoring

#### Run Drift Detection with Evidently

The monitoring pipeline performs:
- Baseline inference on test data
- Drift dataset creation
- Evidently report generation
- Upload reports to S3

```bash
python sagemaker/drift_check.py
```

The script will:
1. Load baseline (training) and current (test/drift) datasets
2. Generate comparison reports with:
   - Data drift metrics
   - Model performance metrics
   - Statistical test results
3. Save HTML and JSON reports to S3

#### View Evidently Reports

Reports are saved to S3 at:
- HTML: `s3://<bucket>/monitoring/reports/evidently_report_<timestamp>.html`
- JSON: `s3://<bucket>/monitoring/reports/evidently_report_<timestamp>.json`

Download and open the HTML file in a browser to view interactive visualizations.

### 6. Endpoint Evaluation

Test the deployed endpoint:

```bash
python src/evaluation/test_endpoint.py
```

## Model Performance

The system aims to significantly outperform the baseline RMSE of $5–$8 through:

1. **Advanced Feature Engineering**: Temporal and geographic features
2. **Automated Model Selection**: SageMaker Autopilot explores multiple algorithms
3. **Hyperparameter Optimization**: Automatic tuning for best performance
4. **Ensemble Methods**: Autopilot can create ensemble models for better accuracy

## 🔍 Monitoring & Drift Detection

### What We Monitor

- **Input Data Drift**: Changes in feature distributions (using KS test for numeric, Chi-square for categorical)
- **Model Performance**: Prediction accuracy and error metrics over time
- **Prediction Distribution**: Changes in prediction patterns

### Evidently AI Features

- **Data Drift Preset**: Automatically detects drift in input features
- **Regression Preset**: Monitors regression-specific metrics (RMSE, MAE, etc.)
- **Interactive Dashboards**: Visual comparison between baseline and current data
- **Statistical Significance**: P-values and test statistics for drift detection

### Alerting

Configure CloudWatch alarms or SNS notifications based on:
- Drift detection thresholds
- Performance degradation
- Endpoint health metrics

## Infrastructure

The project uses AWS services:

- **Amazon S3**: Data storage and model artifacts
- **Amazon SageMaker**: Model training, deployment, and monitoring
- **SageMaker Autopilot**: Automated ML
- **SageMaker Model Registry**: Model versioning and lifecycle management
- **SageMaker Managed MLflow**: Experiment tracking
- **Amazon MWAA** (Optional): Workflow orchestration
- **Evidently AI**: Model monitoring and drift detection

## Key Files

- `src/preprocessing/preprocess.py`: Feature engineering and data cleaning
- `models/baseline.py`: SageMaker Autopilot training with MLflow tracking
- `sagemaker/pipeline.py`: End-to-end pipeline orchestration
- `sagemaker/drift_check.py`: Evidently-based drift detection and monitoring
- `src/inference/deployment.py`: Model deployment automation
- `src/eda/eda.py`: Exploratory data analysis and visualization

## Security & Best Practices

- IAM roles with least privilege access
- S3 bucket encryption at rest
- VPC endpoints for secure SageMaker communication
- Model artifact versioning
- Manual approval workflow for production deployments

## Additional Resources

- [SageMaker Autopilot Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/autopilot-automate-machine-learning.html)
- [Evidently AI Documentation](https://docs.evidentlyai.com/)
- [SageMaker Model Monitor](https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor.html)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)

## Authors

Deepali Dagar, Sakshi Bokil, Savita Murthi, Swayam Desai - Team Epoch

## Acknowledgments

- Dataset: Kaggle NYC Taxi Fare Prediction
- AWS SageMaker team
- Evidently AI for monitoring tools

---


