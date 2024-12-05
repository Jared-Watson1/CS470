import pandas as pd
import numpy as np
import sqlite3
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    KFold,
    cross_val_score,
)
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    make_scorer,
)
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.inspection import permutation_importance  # Import this
import time
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=UserWarning)

start_time = time.time()
print("Connecting to the database...")

cnx = sqlite3.connect("data.sqlite")

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
df = pd.read_sql_query(f"SELECT {','.join(col)} FROM 'Fires'", cnx)
print(f"Data loaded. Shape: {df.shape}\n")

# Preprocessing
print("Preprocessing data...")
t0 = time.time()
df["DISCOVERY_DATETIME"] = pd.to_datetime(
    df["FIRE_YEAR"] * 1000 + df["DISCOVERY_DOY"], format="%Y%j", errors="coerce"
)
df["CONT_DAYS"] = df["CONT_DATE"] - df["DISCOVERY_DATE"]
df["CONT_DAYS"] = df["CONT_DAYS"].astype(float)
df["CONT_DATETIME"] = pd.to_datetime(
    (df["FIRE_YEAR"] + np.floor(df["CONT_DAYS"] / 365)) * 1000 + df["CONT_DOY"],
    format="%Y%j",
    errors="coerce",
)
print(f"Time taken for preprocessing: {time.time() - t0:.2f} seconds\n")


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

df["DAY_OF_WEEK"] = df["DISCOVERY_DATETIME"].dt.dayofweek
df["IS_WEEKEND"] = df["DAY_OF_WEEK"].apply(lambda x: 1 if x >= 5 else 0)

# Create 'Region' Feature using KMeans clustering
coords = df[["LATITUDE", "LONGITUDE"]].dropna()
scaler = StandardScaler()
coords_scaled = scaler.fit_transform(coords)

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
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

# Remove outliers
print("Handling outliers in the target variable...")
t0 = time.time()
upper_limit = df_reg[target].quantile(0.99)
df_reg = df_reg[df_reg[target] <= upper_limit]
print(f"Data after removing outliers. Shape: {df_reg.shape}")
print(f"Time taken to handle outliers: {time.time() - t0:.2f} seconds\n")

print("Applying log transformation to the target variable...")
df_reg["LOG_CONT_DAYS"] = np.log1p(df_reg[target])

sample_size = 50000
print(f"Sampling {sample_size} rows from the data...")
t0 = time.time()
df_reg = df_reg.sample(n=sample_size, random_state=42)
print(f"Time taken to sample data: {time.time() - t0:.2f} seconds\n")

print("Preparing feature matrix X and target vector y...")
X = df_reg[features].copy()
y = df_reg["LOG_CONT_DAYS"]

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

# Feature Scaling
print("Scaling numerical features...")
t0 = time.time()
numerical_features = [
    "FIRE_YEAR",
    "DISCOVERY_DOY",
    "DISCOVERY_HOUR",
    "FIRE_SIZE",
    "LATITUDE",
    "LONGITUDE",
]
scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])
print(f"Time taken for scaling: {time.time() - t0:.2f} seconds\n")

# Add interaction terms
print("Adding interaction terms...")
t0 = time.time()
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X)
feature_names = poly.get_feature_names_out(X.columns)
X = pd.DataFrame(X_poly, columns=feature_names)
print(f"New feature set shape after adding interaction terms: {X.shape}")
print(f"Time taken to add interaction terms: {time.time() - t0:.2f} seconds\n")

# Split the dataset into training and testing sets
print("Splitting data into training and testing sets...")
t0 = time.time()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")
print(f"Time taken to split data: {time.time() - t0:.2f} seconds\n")


# Define a function to evaluate models using cross validation
def evaluate_model_cv(model, X, y, cv=5):
    scorer = make_scorer(mean_squared_error, greater_is_better=False)
    scores = cross_val_score(model, X, y, cv=cv, scoring=scorer, n_jobs=-1)
    mean_score = -np.mean(scores)
    return mean_score


# Random Forest Regression Model with Hyperparameter Tuning
print("Training Random Forest Regression model with hyperparameter tuning...")
t0 = time.time()
rf = RandomForestRegressor(random_state=42, n_jobs=-1)

rf_params = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", "log2"],
}

rf_random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=rf_params,
    n_iter=10,
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1,
)

rf_random_search.fit(X_train, y_train)
rf_best_model = rf_random_search.best_estimator_
print(f"Best parameters: {rf_random_search.best_params_}")
print(f"Time taken to train Random Forest: {time.time() - t0:.2f} seconds")

print("Predicting with Random Forest Regression model...")
t0 = time.time()
y_pred_rf = rf_best_model.predict(X_test)
print(f"Time taken for prediction: {time.time() - t0:.2f} seconds\n")

# HistGradientBoostingRegressor Model with Hyperparameter Tuning
print("Training HistGradientBoostingRegressor model with hyperparameter tuning...")
t0 = time.time()
hgb = HistGradientBoostingRegressor(random_state=42)

hgb_params = {
    "learning_rate": [0.01, 0.1],
    "max_iter": [100, 200],
    "max_depth": [10, 20],
    "min_samples_leaf": [20, 50],
    "max_leaf_nodes": [31, 50],
}

hgb_random_search = RandomizedSearchCV(
    estimator=hgb,
    param_distributions=hgb_params,
    n_iter=10,
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1,
)

hgb_random_search.fit(X_train, y_train)
hgb_best_model = hgb_random_search.best_estimator_
print(f"Best parameters: {hgb_random_search.best_params_}")
print(
    f"Time taken to train HistGradientBoostingRegressor: {time.time() - t0:.2f} seconds"
)

print("Predicting with HistGradientBoostingRegressor model...")
t0 = time.time()
y_pred_hgb = hgb_best_model.predict(X_test)
print(f"Time taken for prediction: {time.time() - t0:.2f} seconds\n")


# Evaluating Models
def evaluate_model(name, y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\n{name} Model Performance:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R^2 Score: {r2:.4f}")


print("Evaluating models...")
t0 = time.time()
evaluate_model("Random Forest Regression", y_test, y_pred_rf)
evaluate_model("HistGradientBoostingRegressor", y_test, y_pred_hgb)
print(f"\nTime taken to evaluate models: {time.time() - t0:.2f} seconds\n")

# Inverse transform predictions for MAE and MSE comparison
y_test_original = np.expm1(y_test)
y_pred_rf_original = np.expm1(y_pred_rf)
y_pred_hgb_original = np.expm1(y_pred_hgb)

print("Evaluating models on original scale...")
evaluate_model("Random Forest Regression", y_test_original, y_pred_rf_original)
evaluate_model("HistGradientBoostingRegressor", y_test_original, y_pred_hgb_original)

# Feature Importance from HistGradientBoostingRegressor
print("Calculating feature importances from HistGradientBoostingRegressor model...")
t0 = time.time()
try:
    importances = hgb_best_model.feature_importances_
except AttributeError:
    print(
        "feature_importances_ attribute not found, computing permutation importances..."
    )
    result = permutation_importance(
        hgb_best_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )
    importances = result.importances_mean
feature_names = X.columns
hgb_importances = pd.Series(importances, index=feature_names)
print(f"Time taken to calculate feature importances: {time.time() - t0:.2f} seconds\n")

# Plot Feature Importances
print("Plotting feature importances...")
t0 = time.time()
plt.figure(figsize=(10, 8))
hgb_importances.nlargest(20).plot(kind="barh")
plt.title("Top 20 Feature Importances from HistGradientBoostingRegressor")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()
print(f"Time taken to plot feature importances: {time.time() - t0:.2f} seconds\n")

# Residual Analysis
print("Performing residual analysis...")
t0 = time.time()
residuals = y_test - y_pred_hgb
sns.histplot(residuals, kde=True)
plt.title("Residual Distribution (Log Scale)")
plt.show()

print(f"Time taken for residual analysis: {time.time() - t0:.2f} seconds\n")

end_time = time.time()
print(f"Total execution time: {end_time - start_time:.2f} seconds")
