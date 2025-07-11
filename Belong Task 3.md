```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Step 1: Load the dataset from CSV
file_path = 'C:\\Users\\vidya\\Downloads\\Flyzy Flight Cancellation - Sheet1.csv'  # Update with your CSV file name
df = pd.read_csv(file_path)

# Step 2: Drop irrelevant columns
df.drop(columns=['Flight ID'], inplace=True)

# Step 3: Separate features and target
X = df.drop('Flight_Cancelled', axis=1)
y = df['Flight_Cancelled']

# Step 4: Identify column types
categorical_cols = ['Airline', 'Origin_Airport', 'Destination_Airport', 'Airplane_Type']
numerical_cols = [col for col in X.columns if col not in categorical_cols]

# Step 5: Define preprocessing steps
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(drop='first'), categorical_cols)
])

# Step 6: Build pipeline with Logistic Regression
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Step 7: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Train model
pipeline.fit(X_train, y_train)

# Step 9: Predict
y_pred = pipeline.predict(X_test)

# Step 10: Evaluate
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Step 11: Print results
print("Model Evaluation Metrics:")
print(f"Accuracy : {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall   : {recall:.2f}")
print(f"F1-score : {f1:.2f}")

```

    Model Evaluation Metrics:
    Accuracy : 0.80
    Precision: 0.83
    Recall   : 0.89
    F1-score : 0.86
    


```python

```
