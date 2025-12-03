import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import os
import argparse

def compare_models(args):
    dyt_path = os.path.join(args.results_dir, 'dyt_results.csv')
    tft_path = os.path.join(args.results_dir, 'tft_results.csv')
    
    if not os.path.exists(dyt_path) or not os.path.exists(tft_path):
        print("Results files not found. Run evaluation first.")
        return
        
    dyt_df = pd.read_csv(dyt_path)
    tft_df = pd.read_csv(tft_path)
    
    # ROC Curve
    fpr_dyt, tpr_dyt, _ = roc_curve(dyt_df['y_true'], dyt_df['y_score'])
    fpr_tft, tpr_tft, _ = roc_curve(tft_df['y_true'], tft_df['y_score'])
    
    auc_dyt = auc(fpr_dyt, tpr_dyt)
    auc_tft = auc(fpr_tft, tpr_tft)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr_dyt, tpr_dyt, label=f'DyT (AUC = {auc_dyt:.3f})')
    plt.plot(fpr_tft, tpr_tft, label=f'TFT (AUC = {auc_tft:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend()
    plt.savefig(os.path.join(args.results_dir, 'roc_comparison.png'))
    print(f"Saved ROC plot to {args.results_dir}/roc_comparison.png")
    
    # PRC Curve
    precision_dyt, recall_dyt, _ = precision_recall_curve(dyt_df['y_true'], dyt_df['y_score'])
    precision_tft, recall_tft, _ = precision_recall_curve(tft_df['y_true'], tft_df['y_score'])
    
    auprc_dyt = auc(recall_dyt, precision_dyt)
    auprc_tft = auc(recall_tft, precision_tft)
    
    plt.figure(figsize=(10, 6))
    plt.plot(recall_dyt, precision_dyt, label=f'DyT (AUPRC = {auprc_dyt:.3f})')
    plt.plot(recall_tft, precision_tft, label=f'TFT (AUPRC = {auprc_tft:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve Comparison')
    plt.legend()
    plt.savefig(os.path.join(args.results_dir, 'prc_comparison.png'))
    print(f"Saved PRC plot to {args.results_dir}/prc_comparison.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='results')
    args = parser.parse_args()
    compare_models(args)
