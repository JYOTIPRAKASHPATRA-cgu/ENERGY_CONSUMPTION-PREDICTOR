import pandas as pd

df = pd.read_csv("small_df.csv")
print(df.head())

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Load dataset
df = pd.read_csv("small_df.csv")

# Encode categorical column
df["Day"] = df["Day"].astype("category").cat.codes

# Features & Target
X = df.drop("Energy", axis=1)
y = df["Energy"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)

print("R2 Score:", r2_score(y_test, preds))
print("MAE:", mean_absolute_error(y_test, preds))

# Save Model
joblib.dump(model, "model.pkl")
print("Model saved successfully ✅")