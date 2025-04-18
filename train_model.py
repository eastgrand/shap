import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load your dataset
df = pd.read_csv("zip_features.csv")

# Set your target variable
target_column = "2024_Bought_Nike_Athletic_Shoes_12_Mo_39_2024 Bought Nike Athletic Shoes Last 12 Mo (%)"
y = df[target_column]
X = df.drop(columns=[target_column])

# Convert object columns to categorical or encode them
for col in X.select_dtypes(include="object").columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# Same for the target if it's categorical
if y.dtype == 'object':
    y = LabelEncoder().fit_transform(y.astype(str))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1)
model.fit(X_train, y_train)

# Save the model using XGBoost's native format
model.save_model("model.json")

print("✅ Model saved as model.json")
