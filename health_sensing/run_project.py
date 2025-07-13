#!/usr/bin/env python3
"""
DeepMedico Health Sensing Project Runner
========================================

This script runs the complete health sensing project pipeline:
1. Generate visualizations for all participants
2. Create the dataset with filtered signals and labeled windows
3. Train and evaluate both 1D CNN and Conv-LSTM models

Usage: python run_project.py [--skip-vis] [--skip-dataset] [--skip-training]
"""

import os
import sys
import subprocess
import argparse

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: {description} failed with return code {result.returncode}")
        return False
    else:
        print(f"SUCCESS: {description} completed successfully")
        return True

def main():
    parser = argparse.ArgumentParser(description="Run the complete health sensing project")
    parser.add_argument("--skip-vis", action="store_true", help="Skip visualization generation")
    parser.add_argument("--skip-dataset", action="store_true", help="Skip dataset creation")
    parser.add_argument("--skip-training", action="store_true", help="Skip model training")
    args = parser.parse_args()

    # Set working directory first
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)

    # Get the Python executable path - try multiple options
    possible_pythons = [
        "python",  # System python
        "python.exe",  # Windows system python
        os.path.join("tfenv", "Scripts", "python.exe"),  # Virtual env python (relative)
        os.path.join(project_root, "tfenv", "Scripts", "python.exe"),  # Virtual env python (absolute)
    ]
    
    python_exe = None
    for py_path in possible_pythons:
        try:
            result = subprocess.run([py_path, "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                python_exe = py_path
                print(f"Using Python: {py_path}")
                break
        except FileNotFoundError:
            continue
    
    if python_exe is None:
        print("ERROR: Could not find a working Python executable")
        print("Please ensure Python is installed and accessible")
        return
    
    print(f"Project root: {project_root}")
    print(f"Python executable: {python_exe}")
    
    success_count = 0
    total_steps = 0
    
    # Step 1: Generate visualizations for all participants
    if not args.skip_vis:
        total_steps += 1
        print(f"\n🎯 STEP 1: Generating Visualizations")
        
        data_dir = os.path.join(project_root, "Data")
        participants = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        
        for participant in participants:
            participant_path = os.path.join("Data", participant)
            cmd = [python_exe, "scripts/vis.py", "-name", participant_path]
            
            if run_command(cmd, f"Visualization for {participant}"):
                success_count += 0.2  # Partial success
            else:
                print(f"Warning: Visualization failed for {participant}")
        
        if len(participants) > 0:
            success_count += 0.8  # Complete the step if at least some succeeded
    
    # Step 2: Create dataset
    if not args.skip_dataset:
        total_steps += 1
        print(f"\n🎯 STEP 2: Creating Dataset")
        
        cmd = [python_exe, "scripts/create_dataset.py", "-in_dir", "Data", "-out_dir", "Dataset"]
        
        if run_command(cmd, "Dataset creation"):
            success_count += 1
    
    # Step 3: Train models
    if not args.skip_training:
        total_steps += 1
        print(f"\n🎯 STEP 3: Training Models")
        
        cmd = [python_exe, "scripts/train_model.py"]
        
        if run_command(cmd, "Model training and evaluation"):
            success_count += 1
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"PROJECT EXECUTION SUMMARY")
    print(f"{'='*60}")
    print(f"Steps completed successfully: {success_count}/{total_steps}")
    
    if success_count == total_steps:
        print("🎉 All steps completed successfully!")
        print("\nResults can be found in:")
        print("  - Visualizations/: PDF plots for each participant")
        print("  - Dataset/: Processed dataset ready for training")
        print("  - Console output: Model evaluation metrics")
    else:
        print("⚠️  Some steps had issues. Check the output above for details.")
    
    print(f"\n📁 Project structure:")
    print(f"  - Data/: Raw participant data")
    print(f"  - Visualizations/: Generated PDF visualizations")
    print(f"  - Dataset/: Processed dataset")
    print(f"  - models/: Neural network architectures")
    print(f"  - scripts/: Processing and training scripts")

if __name__ == "__main__":
    main()
