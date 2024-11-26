import pandas as pd
import numpy as np
import sqlite3
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import time
import warnings
import matplotlib.pyplot as plt
import xgboost as xgb

warnings.filterwarnings("ignore", category=UserWarning)

start_time = time.time()
print("Connecting to the database...")

# Connect to the SQLite database
cnx = sqlite3.connect("data.sqlite")

# Columns to select from the database
col = [
    "FIRE_YEAR",
    "DISCOVERY_DATE",
    "DISCOVERY_DOY",
    "DISCOVERY_TIME",
    "STAT_CAUSE_CODE",
    "STAT_CAUSE_DESCR",
    "CONT_DATE",
    "CONT_DOY",
    "CONT_TIME",
    "FIRE_SIZE",
    "FIRE_SIZE_CLASS",
    "LATITUDE",
    "LONGITUDE",
    "STATE",
    "COUNTY",
    "FIPS_CODE",
    "FIPS_NAME",
]

print("Loading data from the database...")
# Load the data into a DataFrame
df = pd.read_sql_query(f"SELECT {','.join(col)} FROM 'Fires'", cnx)
print(f"Data loaded. Shape: {df.shape}\n")

# Preprocessing
print("Preprocessing data...")
t0 = time.time()
df["DISCOVERY_DATETIME"] = pd.to_datetime(
    df["FIRE_YEAR"] * 1000 + df["DISCOVERY_DOY"], format="%Y%j", errors="coerce"
)
df["CONT_DAYS"] = df["CONT_DATE"] - df["DISCOVERY_DATE"]
df["CONT_DATETIME"] = pd.to_datetime(
    (df["FIRE_YEAR"] + np.floor(df["CONT_DAYS"] / 365)) * 1000 + df["CONT_DOY"],
    format="%Y%j",
    errors="coerce",
)
print(f"Time taken for preprocessing: {time.time() - t0:.2f} seconds\n")


# Function to process 'DISCOVERY_TIME'
def process_discovery_time(time_str):
    if pd.isnull(time_str):
        return np.nan
    else:
        time_str = str(time_str).zfill(4)
        try:
            hour = int(time_str[:2])
            minute = int(time_str[2:])
            return hour + minute / 60.0
        except ValueError:
            return np.nan


print("Processing DISCOVERY_TIME...")
t0 = time.time()
df["DISCOVERY_HOUR"] = df["DISCOVERY_TIME"].apply(process_discovery_time)
print(f"Time taken to process DISCOVERY_TIME: {time.time() - t0:.2f} seconds\n")

# Feature Engineering
print("Feature Engineering...")
t0 = time.time()


# Add 'Season' Feature
def assign_season(doy):
    if doy >= 80 and doy <= 171:
        return "Spring"
    elif doy >= 172 and doy <= 263:
        return "Summer"
    elif doy >= 264 and doy <= 354:
        return "Fall"
    else:
        return "Winter"


df["SEASON"] = df["DISCOVERY_DOY"].apply(assign_season)

# Add 'Is_Weekend' Feature
df["DAY_OF_WEEK"] = df["DISCOVERY_DATETIME"].dt.dayofweek
df["IS_WEEKEND"] = df["DAY_OF_WEEK"].apply(lambda x: 1 if x >= 5 else 0)

# Create 'Region' Feature using KMeans clustering
coords = df[["LATITUDE", "LONGITUDE"]].dropna()
scaler = StandardScaler()
coords_scaled = scaler.fit_transform(coords)

kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(coords_scaled)
df.loc[coords.index, "REGION"] = kmeans.labels_

print(f"Time taken for feature engineering: {time.time() - t0:.2f} seconds\n")

# Select features and target variable
features = [
    "FIRE_YEAR",
    "DISCOVERY_DOY",
    "DISCOVERY_HOUR",
    "STAT_CAUSE_CODE",
    "FIRE_SIZE",
    "FIRE_SIZE_CLASS",
    "LATITUDE",
    "LONGITUDE",
    "STATE",
    "SEASON",
    "IS_WEEKEND",
    "REGION",
]
target = "CONT_DAYS"

# Drop rows with missing values
print("Dropping rows with missing values...")
t0 = time.time()
df_reg = df[features + [target]].dropna()
print(f"Data after dropping missing values. Shape: {df_reg.shape}")
print(f"Time taken to drop missing values: {time.time() - t0:.2f} seconds\n")

# Sample a subset of the data for faster computation
sample_size = 100000  # Adjust sample size as needed
print(f"Sampling {sample_size} rows from the data...")
t0 = time.time()
df_reg = df_reg.sample(n=sample_size, random_state=42)
print(f"Time taken to sample data: {time.time() - t0:.2f} seconds\n")

# Prepare feature matrix X and target vector y
print("Preparing feature matrix X and target vector y...")
X = df_reg[features].copy()
y = df_reg[target]

# Encode categorical variables
print("Encoding categorical variables...")
t0 = time.time()
categorical_features = [
    "STAT_CAUSE_CODE",
    "FIRE_SIZE_CLASS",
    "STATE",
    "SEASON",
    "REGION",
]
for col in categorical_features:
    X[col] = X[col].astype(str)

print("Applying one-hot encoding...")
X = pd.get_dummies(X, columns=categorical_features, drop_first=True)
print(f"Feature set shape after encoding: {X.shape}")
print(f"Time taken for encoding: {time.time() - t0:.2f} seconds\n")

# Split the dataset into training and testing sets
print("Splitting data into training and testing sets...")
t0 = time.time()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")
print(f"Time taken to split data: {time.time() - t0:.2f} seconds\n")


# Define a function to evaluate models
def evaluate_model(name, y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\n{name} Model Performance:")
    print(f"MSE: {mse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R^2 Score: {r2:.4f}")


# Linear Regression Model
print("Training Linear Regression model...")
t0 = time.time()
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
print(f"Time taken to train Linear Regression: {time.time() - t0:.2f} seconds")

print("Predicting with Linear Regression model...")
t0 = time.time()
y_pred_lr = lr_model.predict(X_test)
print(f"Time taken for prediction: {time.time() - t0:.2f} seconds\n")

# Random Forest Regression Model with Hyperparameter Tuning
print("Training Random Forest Regression model with hyperparameter tuning...")
t0 = time.time()
rf = RandomForestRegressor(random_state=42, n_jobs=-1)

rf_params = {
    "n_estimators": [50, 100],
    "max_depth": [10, 20],
    "min_samples_split": [2, 5],
}

rf_random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=rf_params,
    n_iter=5,
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1,
)

rf_random_search.fit(X_train, y_train)
rf_best_model = rf_random_search.best_estimator_
print(f"Time taken to train Random Forest: {time.time() - t0:.2f} seconds")

print("Predicting with Random Forest Regression model...")
t0 = time.time()
y_pred_rf = rf_best_model.predict(X_test)
print(f"Time taken for prediction: {time.time() - t0:.2f} seconds\n")

# XGBoost Regression Model
print("Training XGBoost Regression model...")
t0 = time.time()
xgb_model = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=100,
    max_depth=10,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)
xgb_model.fit(X_train, y_train)
print(f"Time taken to train XGBoost: {time.time() - t0:.2f} seconds")

print("Predicting with XGBoost Regression model...")
t0 = time.time()
y_pred_xgb = xgb_model.predict(X_test)
print(f"Time taken for prediction: {time.time() - t0:.2f} seconds\n")

# Evaluating Models
print("Evaluating models...")
t0 = time.time()
evaluate_model("Linear Regression", y_test, y_pred_lr)
evaluate_model("Random Forest Regression", y_test, y_pred_rf)
evaluate_model("XGBoost Regression", y_test, y_pred_xgb)
print(f"\nTime taken to evaluate models: {time.time() - t0:.2f} seconds\n")

# Feature Importance from XGBoost
print("Calculating feature importances from XGBoost model...")
t0 = time.time()
importances = xgb_model.feature_importances_
feature_names = X_train.columns
xgb_importances = pd.Series(importances, index=feature_names)
print(f"Time taken to calculate feature importances: {time.time() - t0:.2f} seconds\n")

# Plot Feature Importances
print("Plotting feature importances...")
t0 = time.time()
plt.figure(figsize=(10, 8))
xgb_importances.nlargest(20).plot(kind="barh")
plt.title("Top 20 Feature Importances from XGBoost")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()
print(f"Time taken to plot feature importances: {time.time() - t0:.2f} seconds\n")

end_time = time.time()
print(f"Total execution time: {end_time - start_time:.2f} seconds")
