import sys
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import logging
from typing import Dict, List, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def vectorized_higuchi_fd(time_series: np.ndarray, k_max: int) -> float:
    """Calculate the Higuchi Fractal Dimension of the time series using vectorization."""
    N = len(time_series)
    k_range = np.arange(1, min(k_max, N//2) + 1)
    Lk = np.zeros(len(k_range))
    
    for k_idx, k in enumerate(k_range):
        Lmk = np.zeros(k)
        for m in range(k):
            indices = np.arange(1, (N-m)//k)
            Lmki = np.sum(np.abs(time_series[m+indices*k] - time_series[m+(indices-1)*k]))
            Lmki = Lmki * (N - 1) / (((N-m)//k) * k)
            Lmk[m] = Lmki
        Lk[k_idx] = np.mean(Lmk)
    
    x = np.log(1/k_range)
    y = np.log(Lk)
    polyfit = np.polyfit(x, y, 1)
    return polyfit[0]

def vectorized_dfa(time_series: np.ndarray, scale_lim: List[int] = [5, 100]) -> float:
    """Perform vectorized Detrended Fluctuation Analysis (DFA) on the time series."""
    N = len(time_series)
    scales = np.arange(scale_lim[0], min(scale_lim[1], N // 4))
    F = np.zeros(len(scales))
    
    y = np.cumsum(time_series - np.mean(time_series))
    
    for i, scale in enumerate(scales):
        segments = N // scale
        F_scale = np.zeros(segments)
        
        for j in range(segments):
            segment = y[j*scale:(j+1)*scale]
            t = np.arange(scale)
            coef = np.polyfit(t, segment, 1)
            trend = np.polyval(coef, t)
            F_scale[j] = np.sqrt(np.mean((segment - trend)**2))
        
        F[i] = np.mean(F_scale)
    
    log_scales = np.log(scales)
    log_F = np.log(F)
    
    coef = np.polyfit(log_scales, log_F, 1)
    return coef[0]

def analyze_audio(audio_file: str, duration: int = 60, k_max: int = 10, window_sizes: List[float] = [1, 3]) -> Tuple[Dict, int]:
    """Perform multi-scale fractal analysis on an audio file."""
    logging.info(f"Analyzing audio file: {audio_file}")
    try:
        signal, sr = librosa.load(audio_file, sr=None, duration=duration)
        
        results = {'time': [], 'hfd': {size: [] for size in window_sizes}, 'dfa': {size: [] for size in window_sizes}}
        
        for size in window_sizes:
            window_samples = int(size * sr)
            step_samples = int(0.1 * sr)  # 10% overlap
            
            for i in tqdm(range(0, len(signal) - window_samples + 1, step_samples),
                          desc=f"Analyzing (window: {size}s)"):
                end = i + window_samples
                window = signal[i:end]
                
                hfd = vectorized_higuchi_fd(window, k_max)
                dfa_result = vectorized_dfa(window)
                
                if size == window_sizes[0]:
                    results['time'].append(i / sr)
                results['hfd'][size].append(hfd)
                results['dfa'][size].append(dfa_result)
        
        logging.info("Audio analysis completed successfully")
        return results, sr
    except Exception as e:
        logging.error(f"Error during audio analysis: {str(e)}")
        raise

def adaptive_threshold(results: Dict) -> Tuple[float, float]:
    """Calculate adaptive thresholds for HFD and DFA based on the distribution of results."""
    all_hfd = np.concatenate([results['hfd'][size] for size in results['hfd']])
    all_dfa = np.concatenate([results['dfa'][size] for size in results['dfa']])
    
    hfd_threshold = np.mean(all_hfd) + 0.25 * np.std(all_hfd)  # Lowered from 0.5 to 0.25
    dfa_threshold = np.mean(all_dfa) + 0.25 * np.std(all_dfa)  # Lowered from 0.5 to 0.25
    
    logging.info(f"Calculated thresholds - HFD: {hfd_threshold:.4f}, DFA: {dfa_threshold:.4f}")
    return hfd_threshold, dfa_threshold

def classify_voice(results: Dict, hfd_threshold: float, dfa_threshold: float) -> List[bool]:
    """Classify voice segments based on HFD and DFA thresholds."""
    classifications = []
    min_length = min(len(results['hfd'][size]) for size in results['hfd'])
    
    window_size = 5  # Consider 5 segments at a time
    hfd_weight = 0.7  # Give more weight to HFD
    dfa_weight = 0.3
    
    for i in range(min_length - window_size + 1):
        window_hfd = [results['hfd'][size][i:i+window_size] for size in results['hfd']]
        window_dfa = [results['dfa'][size][i:i+window_size] for size in results['dfa']]
        
        hfd_avg = np.mean([np.mean(w) for w in window_hfd])
        dfa_avg = np.mean([np.mean(w) for w in window_dfa])
        
        hfd_var = np.mean([np.std(w) for w in window_hfd])
        dfa_var = np.mean([np.std(w) for w in window_dfa])
        
        # Higher values and higher variability suggest AI-generated voice
        hfd_ai = (hfd_avg > hfd_threshold) or (hfd_var > 0.1)
        dfa_ai = (dfa_avg > dfa_threshold) or (dfa_var > 0.05)
        
        is_ai = (hfd_weight * hfd_ai + dfa_weight * dfa_ai) > 0.5
        classifications.append(is_ai)
    
    # Pad the classifications list to match the original length
    classifications.extend([classifications[-1]] * (window_size - 1))
    
    logging.info(f"Voice classification completed. {sum(classifications)} segments classified as AI-generated.")
    return classifications

def save_results_to_csv(results: Dict, classifications: List[bool], filename: str):
    """Save analysis results to a CSV file."""
    logging.info(f"Saving results to {filename}")
    try:
        min_length = min(len(results['time']), len(classifications),
                         *(len(results['hfd'][size]) for size in results['hfd']),
                         *(len(results['dfa'][size]) for size in results['dfa']))
        
        df = pd.DataFrame({
            'Time': results['time'][:min_length],
            'Classification': classifications[:min_length]
        })
        
        for size in results['hfd']:
            df[f'HFD_{size}s'] = results['hfd'][size][:min_length]
            df[f'DFA_{size}s'] = results['dfa'][size][:min_length]
        
        df.to_csv(filename, index=False)
        logging.info(f"Results successfully saved to {filename}")
    except Exception as e:
        logging.error(f"Error saving results to CSV: {str(e)}")
        raise

def plot_results(results: Dict, hfd_threshold: float, dfa_threshold: float, classifications: List[bool], filename: str):
    """Create and save a visualization of the fractal analysis results."""
    logging.info("Plotting results")
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        min_length = min(len(results['time']), len(classifications),
                         *(len(results['hfd'][size]) for size in results['hfd']),
                         *(len(results['dfa'][size]) for size in results['dfa']))
        
        for size in results['hfd']:
            ax1.plot(results['time'][:min_length], results['hfd'][size][:min_length], label=f'HFD {size}s')
        ax1.axhline(y=hfd_threshold, color='r', linestyle='--', label='HFD Threshold')
        ax1.set_ylabel('Higuchi Fractal Dimension')
        ax1.legend()
        ax1.grid(True)
        
        for size in results['dfa']:
            ax2.plot(results['time'][:min_length], results['dfa'][size][:min_length], label=f'DFA {size}s')
        ax2.axhline(y=dfa_threshold, color='r', linestyle='--', label='DFA Threshold')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Detrended Fluctuation Analysis')
        ax2.legend()
        ax2.grid(True)
        
        for ax in (ax1, ax2):
            for i, cls in enumerate(classifications[:min_length]):
                if cls:
                    ax.axvspan(results['time'][i], results['time'][i+1] if i+1 < min_length else results['time'][min_length-1],
                               alpha=0.2, color='green')
        
        plt.tight_layout()
        plt.savefig(filename)
        logging.info(f"Plot saved as '{filename}'")
    except Exception as e:
        logging.error(f"Error plotting results: {str(e)}")
        raise

def retroactive_analysis(results: Dict, classifications: List[bool]) -> List[Dict]:
    """Perform retroactive analysis on three segments of the audio."""
    logging.info("Performing retroactive analysis")
    min_length = min(len(results['time']), len(classifications),
                     *(len(results['hfd'][size]) for size in results['hfd']),
                     *(len(results['dfa'][size]) for size in results['dfa']))
    
    total_duration = results['time'][min_length - 1]
    segment_duration = total_duration / 3
    
    segments = [
        (0, segment_duration),
        (segment_duration, 2 * segment_duration),
        (2 * segment_duration, total_duration)
    ]
    
    segment_analysis = []
    
    for start, end in segments:
        segment_indices = (np.array(results['time'][:min_length]) >= start) & (np.array(results['time'][:min_length]) < end)
        
        segment_data = {
            'hfd': {size: np.mean(np.array(results['hfd'][size][:min_length])[segment_indices]) for size in results['hfd']},
            'dfa': {size: np.mean(np.array(results['dfa'][size][:min_length])[segment_indices]) for size in results['dfa']},
            'classification': np.mean(np.array(classifications[:min_length])[segment_indices])
        }
        
        segment_analysis.append(segment_data)
    
    logging.info("Retroactive analysis completed")
    return segment_analysis

def main():
    """Main function to run the fractal voice analysis."""
    if len(sys.argv) != 2:
        logging.error("Usage: python analyze.py <audio_file_path>")
        sys.exit(1)

    audio_file = sys.argv[1]
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    logging.info(f"Starting analysis of audio file: {audio_file}")

    try:
        results, sr = analyze_audio(audio_file)
        
        hfd_threshold, dfa_threshold = adaptive_threshold(results)
        classifications = classify_voice(results, hfd_threshold, dfa_threshold)
        
        csv_filename = f"fractal_analysis_results_{base_name}.csv"
        save_results_to_csv(results, classifications, csv_filename)
        
        plot_filename = f"fractal_analysis_result_{base_name}.png"
        plot_results(results, hfd_threshold, dfa_threshold, classifications, plot_filename)
        
        segment_analysis = retroactive_analysis(results, classifications)
        
        logging.info("\nRetroactive Analysis:")
        for i, segment in enumerate(segment_analysis):
            logging.info(f"\nSegment {i+1}:")
            logging.info(f"  Average Classification: {segment['classification']:.2f}")
            for size in segment['hfd']:
                logging.info(f"  HFD {size}s: {segment['hfd'][size]:.4f}")
                logging.info(f"  DFA {size}s: {segment['dfa'][size]:.4f}")

        overall_classification = np.mean(classifications)
        result = 'AI-generated' if overall_classification > 0.5 else 'Human'
        confidence = max(overall_classification, 1-overall_classification)
        logging.info(f"\nOverall Classification: {result} (Confidence: {confidence:.2%})")
        
        logging.info("Analysis completed successfully")
    except Exception as e:
        logging.error(f"An error occurred during analysis: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
