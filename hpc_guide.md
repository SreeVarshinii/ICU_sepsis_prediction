# HPC Execution Guide for Sepsis Prediction Project

This guide outlines the steps to run the training and evaluation pipeline on a shared HPC cluster (e.g., SLURM-based).

## 1. Environment Setup

Before running jobs, ensure the environment is set up.

### 1.1 Load Modules
Most HPC systems use modules. Load Python and CUDA.
```bash
module load python/3.10
module load cuda/11.8
```

### 1.2 Virtual Environment
Create and activate a virtual environment.
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
*Ensure `requirements.txt` includes: `torch`, `pandas`, `numpy`, `scikit-learn`, `tqdm`, `matplotlib`, `pyarrow`.*

## 2. Data Preparation
Ensure data is available on the cluster.
1.  **Transfer Data**: `scp -r data/unified user@hpc:/path/to/project/data/`
2.  **Generate Splits**: Run the split script if not already done.
    ```bash
    python src/data/make_splits.py
    ```

## 3. Training Job Script (SLURM)
Create a submission script `run_train.sh`.

```bash
#!/bin/bash
#SBATCH --job-name=sepsis_train
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

source venv/bin/activate

# Train DyT Model
echo "Training DyT Model..."
python src/train.py --model dyt --epochs 20 --batch_size 64 --save_dir models/dyt_run

# Train TFT Baseline
echo "Training TFT Baseline..."
python src/train.py --model tft --epochs 20 --batch_size 64 --save_dir models/tft_run
```

Submit with: `sbatch run_train.sh`

## 4. Evaluation Job Script
Create `run_eval.sh`.

```bash
#!/bin/bash
#SBATCH --job-name=sepsis_eval
#SBATCH --output=logs/eval_%j.log
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=8G

source venv/bin/activate

# Evaluate DyT
python src/evaluate.py --model dyt --checkpoint models/dyt_run/dyt_best.pth --output_dir results

# Evaluate TFT
python src/evaluate.py --model tft --checkpoint models/tft_run/tft_best.pth --output_dir results

# Generate Comparison Plots
python src/compare.py --results_dir results
```

## 5. Monitoring
- Check logs: `tail -f logs/train_*.log`
- Monitor GPU usage: `nvidia-smi` (if interactive) or `squeue` for job status.

## 6. Results
After completion, download results:
```bash
scp -r user@hpc:/path/to/project/results ./local_results
```
Check `results/roc_comparison.png` and `results/prc_comparison.png`.
