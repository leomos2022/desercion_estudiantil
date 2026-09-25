"""Pre-train model and save for Vercel deployment."""

import joblib
import sys
sys.path.insert(0, '.')

from src.data_generator import StudentDataGenerator
from src.preprocessing import DataPreprocessor
from sklearn.linear_model import LogisticRegression

print("Generating training data...")
dataset = StudentDataGenerator(
    n_students=5000,
    dropout_rate=0.25,
    random_state=42,
).generate()

print("Preprocessing...")
preprocessor = DataPreprocessor(
    scaling_method="standard",
    imputation_method="mean",
    balance_method=None,
)
x_train, _, y_train, _ = preprocessor.fit_transform(
    dataset,
    target_column="desercion",
    test_size=0.2,
)

print("Training model...")
model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
model.fit(x_train, y_train)

print("Saving artifacts...")
joblib.dump(preprocessor, "api/preprocessor.joblib")
joblib.dump(model, "api/model.joblib")

print("Done! Files saved to api/")
print(f"  preprocessor.joblib: {round(__import__('os').path.getsize('api/preprocessor.joblib')/1024/1024, 2)} MB")
print(f"  model.joblib: {round(__import__('os').path.getsize('api/model.joblib')/1024/1024, 2)} MB")