import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print ("--- Loading data ---")
df = pd.read_csv("bnpl_dataset_v2.csv")

df_model = df.copy()

print("--- Step 2: Feature Engineering ---")
#Framing as binary classifcation, good (paid on time) = 0, bad (late/default) = 1
df_model["Is_High_Risk"] = df_model["Repayment_Status"].apply(lambda x: 0 if x == "Paid On Time" else 1)

# Seprate the features X and y Targets 
X = df_model.drop(["Transaction_ID", "Repayment_Status", "Is_High_Risk"], axis=1)
y = df_model["Is_High_Risk"]

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

print("--- Step 3: Splitting and Preprocessing ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

preprocessor = ColumnTransformer(
    transformers= [
        ("num", StandardScaler(), numerical_cols), 
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols)
    ])

print("--- Step 4: Training Multiple Models---")
models = {
    "Logistical Regression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
    "Random Forrest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42), 
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}
metrics_list = []
test_predictions_df = df.loc[X_test.index].copy()
test_predictions_df["Actual_Risk_label"] = y_test

for model_name, model in models.items():
    print(f"Training {model_name}...")
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    test_predictions_df[f'{model_name}_Prediction'] = y_pred
    test_predictions_df[f'{model_name}_Probability'] = np.round(y_prob, 4)
    
    metrics_list.append({
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1_Score': f1_score(y_test, y_pred)
    })

print("--- Step 5: Exporting Data for Tableau ---")
metrics_df = pd.DataFrame(metrics_list)
metrics_df.to_csv('tableau_model_metrics.csv', index=False)
print("Saved 'tableau_model_metrics.csv'")

test_predictions_df.to_csv('tableau_test_predictions.csv', index=False)
print("Saved 'tableau_test_predictions.csv'")

print("\nAll done! You can now load these CSVs into Tableau.")
