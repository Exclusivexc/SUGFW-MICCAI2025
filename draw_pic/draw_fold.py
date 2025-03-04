import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def get_promise12_data():
    # Define sampling ratios and methods
    ratios = [0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 1.0]
    methods = ['Random', 'ProbCover', 'FPS', 'TypiClust', 'CALR', 'ALPS', 'SUGFW (Ours)']
    
    # Data for each method (mean values)
    data = {
        'Random':     [24.62, 40.92, 46.97, 54.24, 83.11, 85.12, 85.70, 85.06],
        'ProbCover': [12.10, 24.29, 19.03, 37.42, 72.70, 71.92, 73.26, 85.06],
        'FPS':       [34.26, 34.49, 52.62, 50.85, 82.10, 83.63, 85.86, 85.06],
        'TypiClust': [21.80, 22.93, 62.31, 47.31, 82.83, 85.74, 84.42, 85.06],
        'CALR':      [19.73, 28.70, 45.46, 58.34, 74.04, 73.80, 74.64, 85.06],
        'ALPS':      [27.54, 25.27, 54.77, 42.70, 81.67, 85.83, 85.96, 85.06],
        'SUGFW (Ours)':     [35.72, 49.44, 53.09, 65.02, 85.33, 86.39, 86.63, 85.06]
    }
    return ratios, methods, data

def get_utah_data():
    # Define sampling ratios and methods
    ratios = [0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 1.0]
    methods = ['Random', 'ProbCover', 'FPS', 'TypiClust', 'CALR', 'ALPS', 'SUGFW (Ours)']
    
    # Data for each method (mean values)
    data = {
        'Random':     [20.03, 42.66, 46.48, 56.14, 70.06, 71.58, 72.52, 76.81],
        'ProbCover': [35.10, 7.19,  20.79, 16.32, 22.75, 33.32, 37.55, 76.81],
        'FPS':       [11.05, 27.85, 23.70, 33.78, 72.20, 73.07, 71.81, 76.81],
        'TypiClust': [33.79, 41.96, 48.51, 57.96, 75.74, 72.34, 75.11, 76.81],
        'CALR':      [40.80, 47.53, 43.46, 39.43, 56.68, 59.01, 56.16, 76.81],
        'ALPS':      [29.47, 51.50, 46.07, 49.41, 72.53, 71.02, 72.02, 76.81],
        'SUGFW (Ours)':     [37.11, 55.13, 63.14, 65.53, 74.74, 76.12, 77.57, 76.81]
    }
    return ratios, methods, data

def plot_datasets():
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Set style
    plt.style.use('bmh')
    colors = sns.color_palette("husl", n_colors=7)
    markers = ['o', 's', '^', 'D', 'v', 'p', '*']
    
    # Get data for both datasets
    promise_ratios, methods, promise_data = get_promise12_data()
    utah_ratios, _, utah_data = get_utah_data()
    
    # Plot Promise12 data (left subplot)
    lines = []
    for method, color, marker in zip(methods, colors, markers):
        line = ax1.plot(promise_ratios, promise_data[method],
                       marker=marker, markersize=8, linewidth=2,
                       color=color, label=method)[0]
        lines.append(line)
    
    # Add final value annotation on y-axis for Promise12
    final_value = promise_data[methods[0]][-1]  # All methods have same final value
    ax1.axhline(y=final_value, color='gray', linestyle='--', alpha=0.5)
    ax1.text(-0.02, final_value, f'{final_value:.2f}', 
             transform=ax1.get_yaxis_transform(),
             ha='right', va='center', fontsize=12)
    
    ax1.set_xscale('log')
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.set_xticks(promise_ratios)
    ax1.set_xticklabels(['1', '2', '3', '5', '10', '15', '20', '100'], fontsize=14)  # Increased tick label size
    ax1.set_xlim([0.008, 1.2])
    ax1.set_xlabel('Sampling Ratio (%)', fontsize=22)  # Increased label size
    ax1.set_ylabel('Dice Similarity Score (%)', fontsize=22)  # Increased label size
    ax1.set_title('Promise12', fontsize=22)  # Increased title size
    ax1.tick_params(axis='y', labelsize=14)  # Increased y-axis tick label size
    
    # Plot UTAH data (right subplot) at 10x x-coordinates
    utah_ratios_scaled = [x for x in utah_ratios]  # Scale x-coordinates by 10
    for method, color, marker in zip(methods, colors, markers):
        ax2.plot(utah_ratios_scaled, utah_data[method],
                marker=marker, markersize=8, linewidth=2,
                color=color)
    
    # Add final value annotation on y-axis for UTAH
    final_value = utah_data[methods[0]][-1]  # All methods have same final value
    ax2.axhline(y=final_value, color='gray', linestyle='--', alpha=0.5)
    ax2.text(-0.02, final_value, f'{final_value:.2f}',
             transform=ax2.get_yaxis_transform(),
             ha='right', va='center', fontsize=12)
    
    ax2.set_xscale('log')
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    ax2.set_xlim([0.008, 1.2])  # Adjusted limits for scaled coordinates
    ax2.set_xticks(utah_ratios_scaled)  # Use scaled positions for ticks
    # But keep original values for labels
    ax2.set_xticklabels(['0.1', '0.2', '0.3', '0.5', '1', '1.5', '2.0', '100'], fontsize=14)  # Increased tick label size
    ax2.set_xlabel('Sampling Ratio (%)', fontsize=22)  # Increased label size
    ax2.set_title('UTAH', fontsize=22)  # Increased title size
    ax2.tick_params(axis='y', labelsize=14)  # Increased y-axis tick label size
    
    # Legend in lower right corner of second subplot with larger font
    ax2.legend(lines, methods, 
              loc='lower right',
              fontsize=16)  # Increased legend font size
    
    plt.tight_layout()
    
    plt.savefig('dice_scores_combined.png', dpi=300, bbox_inches='tight', format='png')
    plt.savefig('dice_scores_combined.svg', dpi=300, bbox_inches='tight', format='svg')
    plt.close()

def main():
    plot_datasets()

if __name__ == "__main__":
    main()