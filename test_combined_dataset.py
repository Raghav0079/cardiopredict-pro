#!/usr/bin/env python3
"""
Simple test script to verify the combined dataset approach
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import pickle


def test_combined_dataset():
    """Test the combined dataset approach"""
    print("🧪 Testing Combined Dataset Approach")
    print("=" * 50)

    try:
        # Load the first dataset (existing)
        df1 = pd.read_csv("Cardio_vascular.csv")
        print(f"✅ Cardio_vascular.csv loaded: {df1.shape}")

        # Load the second dataset (heart statlog)
        df2 = pd.read_csv("heart_statlog_cleveland_hungary_final.csv")
        print(f"✅ heart_statlog_cleveland_hungary_final.csv loaded: {df2.shape}")

        # Rename columns in df2 to match df1 structure
        df2_renamed = df2.copy()
        df2_renamed.columns = [
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "target",
        ]

        # Add missing features to df2 with default values
        df2_renamed["ca"] = 0  # Default value for number of major vessels
        df2_renamed["thal"] = 2  # Default value for thalassemia (2 = normal)

        # Reorder columns to match df1
        df2_renamed = df2_renamed[df1.columns]

        # Combine both datasets
        df_combined = pd.concat([df1, df2_renamed], ignore_index=True)
        print(f"✅ Combined dataset shape: {df_combined.shape}")

        # Remove duplicates
        df_combined.drop_duplicates(inplace=True)
        print(f"✅ After removing duplicates: {df_combined.shape}")

        # Prepare features and target
        X = df_combined.drop("target", axis=1)
        y = df_combined["target"]

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print(f"\n📊 Dataset Split:")
        print(f"   Training: {X_train.shape[0]} samples")
        print(f"   Testing: {X_test.shape[0]} samples")
        print(f"   Features: {X_train.shape[1]}")
        print(f"   Target distribution: {y.value_counts().to_dict()}")

        # Train and test models
        models = {
            "Logistic Regression": LogisticRegression(
                random_state=42, solver="liblinear"
            ),
            "Random Forest": RandomForestClassifier(
                random_state=42, n_estimators=50
            ),  # Reduced for speed
        }

        results = {}

        for model_name, model in models.items():
            print(f"\n🤖 Training {model_name}...")

            # Train
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            f1 = f1_score(y_test, y_pred)

            results[model_name] = {
                "accuracy": accuracy,
                "roc_auc": roc_auc,
                "f1_score": f1,
            }

            print(f"   Accuracy: {accuracy:.3f}")
            print(f"   ROC AUC: {roc_auc:.3f}")
            print(f"   F1-Score: {f1:.3f}")

        # Save results
        test_results = {
            "dataset_info": {
                "original_samples": len(df1),
                "additional_samples": len(df2_renamed),
                "combined_samples": len(df_combined),
                "improvement": f"{((len(df_combined) - len(df1)) / len(df1) * 100):.1f}%",
            },
            "model_results": results,
            "feature_names": X.columns.tolist(),
        }

        with open("test_results_combined_dataset.pkl", "wb") as f:
            pickle.dump(test_results, f)

        print(f"\n🎯 SUMMARY:")
        print(f"   Original dataset: {len(df1)} samples")
        print(f"   Added dataset: {len(df2_renamed)} samples")
        print(f"   Combined total: {len(df_combined)} samples")
        print(f"   Data improvement: {test_results['dataset_info']['improvement']}")

        # Find best model
        best_model = max(results.items(), key=lambda x: x[1]["f1_score"])
        print(f"   Best model: {best_model[0]} (F1: {best_model[1]['f1_score']:.3f})")

        print("\n✅ Test completed successfully!")
        return test_results

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = test_combined_dataset()
    if results:
        print("\n✅ Combined dataset approach is working correctly!")
    else:
        print("\n❌ There was an issue with the combined dataset approach.")
