import os
import subprocess
import sys

def run_script(script_name):
    print(f"\n{'='*60}")
    print(f"Executing: {script_name}")
    print(f"{'='*60}\n")
    try:
        subprocess.run([sys.executable, script_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Pipeline failed during '{script_name}'.")
        print(f"Error details: {e}")
        sys.exit(1)

def main():
    print("""
=========================================================
   PIPELINE 1: TRAINING & EVALUATION AUTOMATION
=========================================================
This pipeline will:
 1. Clean and prepare the massive dataset.
 2. Train the XGBoost Mathematical Model.
 3. Train the LSTM Deep Learning Sequence Model.
 4. Run the final Hybrid Evaluation to determine the Best Model!
""")
    
    print("\nInitiating Training Pipeline...\n")
    
    # We execute them in strict sequential order
    pipeline_scripts = [
        "prepare_train_test_data.py",
        "train_evaluate_xgboost.py",
        "train_evaluate_lstm.py",
        "hybrid_prediction_final.py",
        "plot_result_visualisation.py"
    ]
    
    for script in pipeline_scripts:
        if os.path.exists(script):
            run_script(script)
        else:
            print(f"[ERROR] Required script '{script}' not found in current directory!")
            sys.exit(1)
            
    print("\nTRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("XGBoost and LSTM are fully trained and locked.")
    print("\nTo evaluate the models and predict on new data, run:")
    print(" -> python pipeline_2_batch_inference.py")

if __name__ == "__main__":
    main()
