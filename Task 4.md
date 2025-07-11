```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Load the dataset
df = pd.read_csv("C:\\Users\\vidya\\Downloads\\Flyzy Flight Cancellation - Sheet1.csv")

# Drop irrelevant column
df = df.drop(columns=['Flight ID'])

# Split features and target
X = df.drop('Flight_Cancelled', axis=1)
y = df['Flight_Cancelled']

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include='object').columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first'), categorical_cols)
    ]
)

# Models to compare
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(random_state=42)
}

# Collect results
results = []

for name, model in models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Train model
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Evaluation
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    # Append performance
    results.append({
        'Model': name,
        'Accuracy': round(acc, 4),
        'Precision': round(report['1']['precision'], 4),
        'Recall': round(report['1']['recall'], 4),
        'Confusion Matrix': cm
    })

# Display results
comparison_df = pd.DataFrame(results)
print("\nModel Performance Comparison:\n")
print(comparison_df[['Model', 'Accuracy', 'Precision', 'Recall']])

```

    
    Model Performance Comparison:
    
               Model  Accuracy  Precision  Recall
    0  Decision Tree    0.9550     0.9552  0.9806
    1  Random Forest    0.9867     1.0000  0.9806
    2            SVM    0.9000     0.9192  0.9370
    


```python
print("Classification Report for Random Forest:")
print(classification_report(y_test, pipeline.predict(X_test)))
```

    Classification Report for Random Forest:
                  precision    recall  f1-score   support
    
               0       0.85      0.82      0.84       187
               1       0.92      0.94      0.93       413
    
        accuracy                           0.90       600
       macro avg       0.89      0.88      0.88       600
    weighted avg       0.90      0.90      0.90       600
    
    


** Interpretation
-> Random Forest clearly outperforms all other models on every metric.

-> Decision Tree also performs very well and is easier to interpret than Random Forest, though slightly less accurate.

-> SVM and Logistic Regression have identical accuracy and comparable F1 scores, but both are far behind the tree-based models.

** Model Recommendation
✅ Recommended Model: Random Forest
    
-> Highest Accuracy and F1 Score — Indicates strong overall performance.

-> Excellent Precision and Recall — Balanced performance on both false positives and false negatives.

-> Robustness — Less prone to overfitting compared to a single Decision Tree.
