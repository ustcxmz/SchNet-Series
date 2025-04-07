import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import seaborn as sns
import pandas as pd
import os
from tqdm import tqdm

# Import the three models (assuming they're in separate files)
from SchNet_on_water_molecule_dataset import SchNet, WaterDataset as WaterDataset1
from SchNet_plus_on_water_molecule_dataset import SchNetPlus, WaterDataset as WaterDataset2
from SchNet_plus_plus_on_water_molecule_dataset import SchNetPlusPlus, WaterDataset as WaterDataset3, collate_fn

def load_datasets(num_samples=200):
    """Load datasets from all three implementations"""
    dataset1 = WaterDataset1(num_samples=num_samples)
    dataset2 = WaterDataset2(num_samples=num_samples)
    dataset3 = WaterDataset3(train=True, test_size=0.2, random_state=42)
    
    # Create consistent test loaders
    test_loader1 = torch.utils.data.DataLoader(dataset1, batch_size=32, collate_fn=collate_fn)
    test_loader2 = torch.utils.data.DataLoader(dataset2, batch_size=32, collate_fn=collate_fn)
    test_loader3 = torch.utils.data.DataLoader(dataset3, batch_size=32, collate_fn=collate_fn)
    
    return test_loader1, test_loader2, test_loader3

def evaluate_model(model, loader, model_type="schnet"):
    """Evaluate a single model with consistent metrics"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_targets = []
    force_cosine_sims = []
    
    with torch.no_grad():
        for batch in loader:
            if model_type in ["schnet", "schnet_plus"]:
                batch = batch.to(device)
                pred = model(batch)
                target = batch.y.view(-1)
                
                # Calculate force cosine similarity (if available)
                if hasattr(batch, 'force'):
                    force_pred = batch.force
                    force_true = batch.force
                    dot_product = torch.sum(force_pred * force_true, dim=-1)
                    norm_pred = torch.norm(force_pred, dim=-1)
                    norm_true = torch.norm(force_true, dim=-1)
                    cosine_sim = dot_product / (norm_pred * norm_true + 1e-8)
                    force_cosine_sims.append(cosine_sim.mean().item())
                
            elif model_type == "schnet_plus_plus":
                z, pos, batch_idx, energy = batch
                z, pos, batch_idx, energy = z.to(device), pos.to(device), batch_idx.to(device), energy.to(device)
                pred = model(z, pos, batch_idx)
                target = energy
            
            all_preds.append(pred.cpu().numpy())
            all_targets.append(target.cpu().numpy())
    
    y_pred = np.concatenate(all_preds).ravel()
    y_true = np.concatenate(all_targets).ravel()
    
    metrics = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred),
        'Pearson': pearsonr(y_true, y_pred)[0],
    }
    
    if force_cosine_sims:
        metrics['Force_Cosine_Similarity'] = np.mean(force_cosine_sims)
    
    return metrics, y_true, y_pred

def plot_comparison(results):
    """Plot comparison of all models"""
    # Convert results to DataFrame for easier plotting
    df = pd.DataFrame(results).T
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16)
    
    # Plot metrics
    metrics = ['MAE', 'RMSE', 'R2', 'Pearson', 'Force_Cosine_Similarity']
    for i, metric in enumerate(metrics):
        if metric in df.columns:
            ax = axes.flatten()[i]
            df[metric].plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            ax.set_title(metric)
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()

def plot_error_distributions(all_errors):
    """Plot error distributions for all models"""
    plt.figure(figsize=(10, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for (model_name, errors), color in zip(all_errors.items(), colors):
        sns.kdeplot(errors, label=model_name, color=color, fill=True, alpha=0.2)
    
    plt.xlabel('Absolute Error (eV)')
    plt.ylabel('Density')
    plt.title('Error Distribution Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('error_distribution_comparison.png')
    plt.close()

def plot_true_vs_pred(all_results):
    """Plot true vs predicted for all models"""
    plt.figure(figsize=(10, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for (model_name, (_, y_true, y_pred)), color in zip(all_results.items(), colors):
        plt.scatter(y_true, y_pred, alpha=0.5, label=model_name, color=color)
    
    # Plot perfect prediction line
    min_val = min(np.min(y_true) for _, y_true, _ in all_results.values())
    max_val = max(np.max(y_true) for _, y_true, _ in all_results.values())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('True vs Predicted Values')
    plt.legend()
    plt.grid(True)
    plt.savefig('true_vs_pred_comparison.png')
    plt.close()

def plot_loss_curves():
    """Plot training loss curves if available"""
    loss_files = {
        'SchNet': 'loss_curve.png',
        'SchNet+': 'loss_curve_plus.png',
        'SchNet++': 'loss_curve_plusplus.png'
    }
    
    plt.figure(figsize=(10, 6))
    
    for model_name, file in loss_files.items():
        if os.path.exists(file):
            img = plt.imread(file)
            plt.imshow(img)
            plt.axis('off')
            plt.title(f'{model_name} Training Curves')
            plt.savefig(f'{model_name.lower()}_loss_curve_standalone.png')
            plt.close()
    
    # Create a composite plot if all files exist
    if all(os.path.exists(f) for f in loss_files.values()):
        plt.figure(figsize=(15, 5))
        for i, (model_name, file) in enumerate(loss_files.items(), 1):
            plt.subplot(1, 3, i)
            img = plt.imread(file)
            plt.imshow(img)
            plt.axis('off')
            plt.title(model_name)
        plt.tight_layout()
        plt.savefig('composite_loss_curves.png')
        plt.close()

def main():
    # Load models (assuming they're saved with standard names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize models
    model1 = SchNet(cutoff=5.0).to(device)
    model1.load_state_dict(torch.load("best_model.pth"))
    
    model2 = SchNetPlus(cutoff=5.0).to(device)
    model2.load_state_dict(torch.load("best_model_plus.pth"))
    
    model3 = SchNetPlusPlus().to(device)
    model3.load_state_dict(torch.load("best_model_plusplus.pth"))
    
    # Load datasets
    test_loader1, test_loader2, test_loader3 = load_datasets()
    
    # Evaluate all models
    results = {}
    all_errors = {}
    
    print("Evaluating SchNet...")
    metrics1, y_true1, y_pred1 = evaluate_model(model1, test_loader1, "schnet")
    results['SchNet'] = metrics1
    all_errors['SchNet'] = np.abs(y_true1 - y_pred1)
    
    print("Evaluating SchNet+...")
    metrics2, y_true2, y_pred2 = evaluate_model(model2, test_loader2, "schnet_plus")
    results['SchNet+'] = metrics2
    all_errors['SchNet+'] = np.abs(y_true2 - y_pred2)
    
    print("Evaluating SchNet++...")
    metrics3, y_true3, y_pred3 = evaluate_model(model3, test_loader3, "schnet_plus_plus")
    results['SchNet++'] = metrics3
    all_errors['SchNet++'] = np.abs(y_true3 - y_pred3)
    
    # Print results
    print("\nModel Comparison Results:")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")
    
    # Generate comparison plots
    plot_comparison(results)
    plot_error_distributions(all_errors)
    
    # Create combined results for true vs pred plot
    all_results = {
        'SchNet': (results['SchNet'], y_true1, y_pred1),
        'SchNet+': (results['SchNet+'], y_true2, y_pred2),
        'SchNet++': (results['SchNet++'], y_true3, y_pred3)
    }
    plot_true_vs_pred(all_results)
    
    # Plot loss curves if available
    plot_loss_curves()
    
    print("\nVisualizations saved to disk.")

if __name__ == "__main__":
    main()