💧 Water Quality Index Prediction with IoT and ML
📌 Overview

This project demonstrates a synthetic IoT-based water quality monitoring platform combined with machine learning to predict the Water Quality Index (WQI). It simulates environmental sensor data from >1,000 locations and trains a model to estimate water quality for real-time decision support.

⚙️ Features

Generate synthetic IoT sensor readings for multiple water quality parameters (pH, DO, BOD, TDS, turbidity, conductivity, etc.)

Compute synthetic WQI using weighted sub-index calculations

Train a Random Forest regression model to predict WQI

Evaluate model performance (MAE, RMSE, R²)

Output:

water_quality_iot.csv — dataset

wqi_model_random_forest.joblib — trained model

metrics.json — evaluation results

feature_importance.png — feature importance plot

residuals_plot.png — residual errors plot

📊 Parameters Used
Parameter	Description	Typical Range
pH	Acidity/alkalinity	6.0 – 9.0
DO (mg/L)	Dissolved Oxygen	1.0 – 12.0
BOD (mg/L)	Biological Oxygen Demand	0.5 – 30.0
TDS (mg/L)	Total Dissolved Solids	50 – 1200
Turbidity (NTU)	Cloudiness of water	0.1 – 150
Conductivity	Electrical conductivity	50 – 2000
Nitrate (mg/L)	Nutrient pollutant	0.1 – 50
Ammonia (mg/L)	Toxic nitrogen compound	0.01 – 10
ORP (mV)	Oxidation-Reduction Potential	-150 – 450
📂 Project Structure
water_quality_iot_ml.py
outputs/
 ├─ water_quality_iot.csv
 ├─ wqi_model_random_forest.joblib
 ├─ metrics.json
 ├─ feature_importance.png
 └─ residuals_plot.png
README.md

🚀 Usage
1. Install Dependencies
pip install numpy pandas scikit-learn matplotlib joblib

2. Run the Script
python water_quality_iot_ml.py --n 1500 --seed 42

3. View Outputs

Go to the outputs/ folder.

Check the dataset CSV, evaluation metrics JSON, trained model file, and plots.

📈 Evaluation Metrics

The script reports:

MAE – Mean Absolute Error

RMSE – Root Mean Squared Error

R² Score – Coefficient of Determination

These metrics are saved in metrics.json.

📌 Future Extensions

Integration with real IoT sensor feeds

Real-time streaming architecture with dashboards (Streamlit/FastAPI)

Multivariate classification for WQI status categories

Cloud-based deployment for live monitoring

📜 License

MIT License — free to use, modify, and distribute with attribution.

📧 Contact

Author: Amos Meremu Dogiye
Github: @Dogiye12
