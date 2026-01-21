#!/usr/bin/env python3
"""
Quick script to check saved model metrics
"""
import pickle

try:
    with open("gradio_trained_models.pkl", "rb") as f:
        data = pickle.load(f)

    print("📊 Saved Model Performance Metrics:")
    print("=" * 50)

    if "metrics" in data:
        for model_name, metrics in data["metrics"].items():
            print(f"\n🤖 {model_name}:")
            print(
                f"   Accuracy: {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)"
            )
            print(
                f"   ROC AUC:  {metrics['roc_auc']:.3f} ({metrics['roc_auc']*100:.1f}%)"
            )
            print(
                f"   F1-Score: {metrics['f1_score']:.3f} ({metrics['f1_score']*100:.1f}%)"
            )

    print("\n" + "=" * 50)
    print("✅ Model metrics retrieved successfully!")

except FileNotFoundError:
    print("❌ No saved model file found")
except Exception as e:
    print(f"❌ Error reading model file: {e}")
