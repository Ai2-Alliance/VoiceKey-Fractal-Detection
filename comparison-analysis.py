import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def load_csv(file_path):
    return pd.read_csv(file_path)

def plot_comparison(df1, df2, measure, window_size, output_file, label1, label2):
    plt.figure(figsize=(12, 6))
    plt.plot(df1['Time'], df1[f'{measure}_{window_size}'], label=label1, alpha=0.7)
    plt.plot(df2['Time'], df2[f'{measure}_{window_size}'], label=label2, alpha=0.7)
    plt.xlabel('Time (s)')
    plt.ylabel(measure)
    plt.title(f'{measure} Comparison ({window_size} window)')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()
    print(f"Comparison plot saved as '{output_file}'")

def plot_classification_comparison(df1, df2, output_file, label1, label2):
    plt.figure(figsize=(12, 6))
    plt.plot(df1['Time'], df1['Classification'], label=label1, alpha=0.7)
    plt.plot(df2['Time'], df2['Classification'], label=label2, alpha=0.7)
    plt.xlabel('Time (s)')
    plt.ylabel('Classification')
    plt.title('Classification Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()
    print(f"Classification comparison plot saved as '{output_file}'")

def calculate_statistics(df, measure, window_size):
    data = df[f'{measure}_{window_size}']
    return {
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data)
    }

def compare_files(file1, file2):
    df1 = load_csv(file1)
    df2 = load_csv(file2)

    label1 = os.path.basename(file1)
    label2 = os.path.basename(file2)

    # Trim to the shorter length
    min_length = min(len(df1), len(df2))
    df1 = df1.iloc[:min_length]
    df2 = df2.iloc[:min_length]

    comparison_data = []

    for measure in ['HFD', 'DFA']:
        for window_size in ['1s', '3s']:
            plot_comparison(df1, df2, measure, window_size, f'{measure}_{window_size}_comparison.png', label1, label2)
            
            stats1 = calculate_statistics(df1, measure, window_size)
            stats2 = calculate_statistics(df2, measure, window_size)
            
            comparison_data.append({
                'Measure': measure,
                'Window': window_size,
                'File': label1,
                **stats1
            })
            comparison_data.append({
                'Measure': measure,
                'Window': window_size,
                'File': label2,
                **stats2
            })

    plot_classification_comparison(df1, df2, 'classification_comparison.png', label1, label2)

    classification_match = (df1['Classification'] == df2['Classification']).mean() * 100

    comparison_df = pd.DataFrame(comparison_data)
    return classification_match, comparison_df, label1, label2

def main():
    if len(sys.argv) != 3:
        print("Usage: python comparison-analysis.py <csv_file_1> <csv_file_2>")
        sys.exit(1)

    file1, file2 = sys.argv[1], sys.argv[2]
    classification_match, comparison_df, label1, label2 = compare_files(file1, file2)

    print(f"Comparing {label1} and {label2}")
    print(f"Classification match: {classification_match:.2f}%")

    # Save detailed comparison to CSV
    comparison_df.to_csv('detailed_comparison.csv', index=False)
    print("Detailed comparison saved to 'detailed_comparison.csv'")

    # Create a summary chart
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Measure', y='mean', hue='File', data=comparison_df)
    plt.title('Mean Fractal Measures Comparison')
    plt.savefig('summary_comparison.png')
    plt.close()
    print("Summary comparison chart saved as 'summary_comparison.png'")

if __name__ == "__main__":
    main()
