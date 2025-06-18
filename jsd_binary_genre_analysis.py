import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import math
from scipy.spatial.distance import jensenshannon

# Path to the dataset
DATASET_PATH = "sampled_dataset_PE_C_fd.csv"

# List of entropy and complexity columns
ENTROPY_COLS = [
    'entropy_amplitude', 'complexity_amplitude',
    'entropy_flux', 'complexity_flux',
    'entropy_harmony', 'complexity_harmony',
    'entropy_spectral', 'complexity_spectral'
]

# List of fractal dimension columns
FRACTAL_COLS = ['higuchi_fd', 'box_counting_fd']

# All metrics to analyze
ALL_METRICS = ENTROPY_COLS + FRACTAL_COLS

# Column for genre
GENRE_COL = 'track_genre'

# Directory for saving results
RESULTS_DIR = 'plots/jsd'
PLOTS_DIR = os.path.join(RESULTS_DIR, 'distributions')
HEATMAPS_DIR = os.path.join(RESULTS_DIR, 'overall_heatmaps')
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(HEATMAPS_DIR, exist_ok=True)


def compute_jsd(p, q):
    """
    Compute Jensen-Shannon Divergence between two probability distributions.
    Both p and q must be 1D numpy arrays and sum to 1.
    Returns the JSD (in bits).
    """
    # Add a small epsilon to avoid log(0)
    eps = 1e-12
    p = np.asarray(p) + eps
    q = np.asarray(q) + eps
    p = p / p.sum()
    q = q / q.sum()
    
    # Use scipy's built-in jensenshannon function
    return jensenshannon(p, q, base=2)


def analyze_jsd_by_genre(df, metrics, genre_column=GENRE_COL, n_bins=30, plot_histograms=True):
    genres = df[genre_column].unique()
    results = []
    # For each metric
    for metric in metrics:
        metric_data = df[[genre_column, metric]].dropna()
        # Compute global min/max for consistent binning
        min_val = metric_data[metric].min()
        max_val = metric_data[metric].max()
        bins = np.linspace(min_val, max_val, n_bins + 1)
        # For each genre
        for genre in genres:
            group1 = metric_data[metric_data[genre_column] == genre][metric]
            group2 = metric_data[metric_data[genre_column] != genre][metric]
            # Compute histograms (probability distributions)
            hist1, _ = np.histogram(group1, bins=bins, density=True)
            hist2, _ = np.histogram(group2, bins=bins, density=True)
            # Normalize to sum to 1
            p = hist1 / (hist1.sum() + 1e-12)
            q = hist2 / (hist2.sum() + 1e-12)
            jsd = compute_jsd(p, q)
            results.append({
                'metric': metric,
                'genre': genre,
                'jsd': jsd,
                'mean_group1': group1.mean(),
                'mean_group2': group2.mean(),
                'std_group1': group1.std(),
                'std_group2': group2.std(),
                'n_group1': len(group1),
                'n_group2': len(group2)
            })
            # Plot distributions
            if plot_histograms:
                plt.figure(figsize=(10, 6))
                sns.histplot(group1, bins=bins, color='blue', label=f'{genre}', stat='density', kde=True, alpha=0.6)
                sns.histplot(group2, bins=bins, color='orange', label='Other genres', stat='density', kde=True, alpha=0.6)
                plt.axvline(group1.mean(), color='blue', linestyle='dashed', label=f'{genre} mean')
                plt.axvline(group2.mean(), color='orange', linestyle='dashed', label='Other genres mean')
                plt.xlabel(metric)
                plt.ylabel('Density')
                plt.title(f'Distribution of {metric} for {genre} vs Other Genres\nJSD={jsd:.3f}')
                plt.legend()
                plot_dir = os.path.join(PLOTS_DIR, metric)
                os.makedirs(plot_dir, exist_ok=True)
                plt.savefig(os.path.join(plot_dir, f'distribution_genre_{genre}.png'), dpi=300, bbox_inches='tight')
                plt.close()
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_DIR, 'jsd_results.csv'), index=False)
    # Create heatmap of JSD values (genres x metrics)
    heatmap_data = results_df.pivot(index='genre', columns='metric', values='jsd').reindex(columns=metrics)
    # Transpose the data to have genres on x-axis and metrics on y-axis
    heatmap_data = heatmap_data.T
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        vmin=0, vmax=1
    )
    plt.title('Jensen-Shannon Divergence (JSD) Heatmap')
    plt.xlabel('Genres')
    plt.ylabel('Metrics')
    plt.tight_layout()
    plt.savefig(os.path.join(HEATMAPS_DIR, 'jsd_heatmap_all_genres.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"JSD analysis complete. Results saved to {RESULTS_DIR}")


def main():
    print("Loading data...")
    df = pd.read_csv(DATASET_PATH)
    print("Performing JSD analysis for all metrics and genres...")
    analyze_jsd_by_genre(df, ALL_METRICS, plot_histograms=True)
    print("Done! Plots and results saved in 'plots/binary/jsd'")


if __name__ == "__main__":
    main() 