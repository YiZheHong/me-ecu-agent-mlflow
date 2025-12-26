"""
Export MLflow Model for Docker Deployment

Simple script to export the latest ECUAgent model from MLflow Registry.

Usage:
    python export_model.py
"""

import mlflow
import shutil
from pathlib import Path

# Configuration
MLFLOW_TRACKING_URI = "http://localhost:5000"
MODEL_NAME = "ECUAgent"
EXPORT_DIR = "./model_export"


def export_model():
    """Export model from MLflow Registry to local directory."""
    print("="*80)
    print("📦 Exporting ECUAgent Model for Docker")
    print("="*80)
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Clean up old export
    export_path = Path(EXPORT_DIR)
    if export_path.exists():
        print(f"\n🗑️  Removing old export: {EXPORT_DIR}")
        shutil.rmtree(export_path)
    
    # Create export directory
    export_path.mkdir(parents=True, exist_ok=True)
    
    # Download model
    print(f"\n⬇️  Downloading model: models:/{MODEL_NAME}/latest")
    print(f"   Destination: {EXPORT_DIR}")
    
    model_uri = f"models:/{MODEL_NAME}/latest"
    
    # Use MLflow's download artifacts
    from mlflow.artifacts import download_artifacts
    
    download_artifacts(
        artifact_uri=model_uri,
        dst_path=EXPORT_DIR
    )
    
    print(f"\n✅ Model exported successfully!")
    print(f"   Location: {export_path.absolute()}")
    print("="*80)
    
    # Verify export
    print("\n📋 Exported files:")
    for item in sorted(export_path.rglob("*")):
        if item.is_file():
            size = item.stat().st_size / (1024*1024)  # MB
            print(f"   {item.relative_to(export_path)} ({size:.2f} MB)")
    
    print("\n✅ Ready for Docker build!")
    print("   Next step: docker build -t ecu-agent .")


if __name__ == "__main__":
    export_model()