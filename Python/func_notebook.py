from importlib.metadata import metadata
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
import os
import uproot
import awkward
import re
from pathlib import Path
import math
import seaborn as sns
import gc
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import pyarrow as py
import multiprocessing
import tempfile
from collections import defaultdict




#Sample window for 64CH digitizer 
baseline_start_64 = 0 
baseline_end_64 = 100
peak_start_64 = 565
peak_end_64 = 578


#Sample window for 4CH digitizer
baseline_start_4 = 0
baseline_end_4 = 50
peak_start_4 = 270
peak_end_4 = 390

#Folders where the data will be stored
parquet_folder = "/home/uzh/hekinc/Master/Python/parquets"
plot_folder = "/home/uzh/hekinc/Master/Python/plots"
comparisons_folder = "/home/uzh/hekinc/Master/Python/plots/comparisons"

####################################################################
#ROOT DATA 
####################################################################

'''
def analyse_root_data_static(filepath, parquet_folder=None, device=None, suffix=None, apply_3sigma_cut=True):
    analyzed_data = [] # List to hold our chunks

    if device is not None:
        baseline_start, baseline_end = baseline_start_4, baseline_end_4
        peak_start, peak_end = peak_start_4, peak_end_4
    else:
        baseline_start, baseline_end = baseline_start_64, baseline_end_64
        peak_start, peak_end = peak_start_64, peak_end_64

    # 1. PROCESS IN CHUNKS (e.g., 100 MB of data at a time)
    for data_dict in uproot.iterate(f"{filepath}:Events", filter_name="/^ch[0-9]+$/", library="np", step_size="100 MB"):
        
        chunk_results = []
        for channel_name, waveforms in data_dict.items():
            print(f"Processing chunk for {channel_name}...")
            
            if waveforms.dtype == 'O': 
                waveforms = np.vstack(waveforms)

            baseline_window = waveforms[:, baseline_start:baseline_end] 
            baselines = np.mean(baseline_window, axis=1)
            std_devs = np.std(baseline_window, axis=1)
                
            corrected_waveforms = waveforms - baselines[:, None]
            corrected_peak_window = corrected_waveforms[:, peak_start:peak_end]
            
            # Find negative peaks specifically (minimum values)
            peaks = np.argmin(corrected_peak_window, axis=1) + peak_start
            peak_values = corrected_waveforms[np.arange(len(waveforms)), peaks]
            
            # 3-Sigma Cut toggle
            if apply_3sigma_cut:
                valid_mask = peak_values < (-3 * std_devs)
            else:
                valid_mask = np.ones(len(peak_values), dtype=bool)
            
            # Find where the signal is negative strictly inside the peak window
            # and only integrate that specific area (between baseline and the curve)
            negative_mask = corrected_peak_window < 0
            integrals = np.sum(corrected_peak_window * negative_mask, axis=1)

            # 2. DROP THE RAW WAVEFORM & REJECT INVALID SIGNALS
            # Only save the calculated features for signals passing the cut
            chunk_results.append(pd.DataFrame({
                "channel": channel_name,
                "baseline": baselines[valid_mask],
                "std_dev": std_devs[valid_mask],
                "peak_integral": integrals[valid_mask]
            }))

            # 3. MANUALLY FREE MEMORY
            del waveforms, corrected_waveforms, corrected_peak_window
            del negative_mask, valid_mask
            gc.collect()

        analyzed_data.append(pd.concat(chunk_results, ignore_index=True))
        
    master_df = pd.concat(analyzed_data, ignore_index=True)
    
    original_name = Path(filepath).parent.name
    if suffix is not None:
        original_name += f"_{suffix}"
        
    final_parquet_path = Path(parquet_folder) / f"{original_name}.parquet"
    final_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the combined DataFrame
    master_df.to_parquet(final_parquet_path, index=False)
    print(f"Data successfully saved to {final_parquet_path}")
        
    return master_df, final_parquet_path
'''





def analyse_root_data(filepath, parquet_folder=None, suffix=None, apply_3sigma_cut=True):
    # Determine integration windows based on device
    
    baseline_start, baseline_end = baseline_start_64, baseline_end_64
    peak_start, peak_end = peak_start_64, peak_end_64

    # Setup output path
    original_name = Path(filepath).parent.name
    if suffix is not None:
        original_name += f"_{suffix}"
        
    final_parquet_path = Path(parquet_folder) / f"{original_name}.parquet"
    final_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    
    pq_writer = None

    # 1. PROCESS IN CHUNKS
    for data_dict in uproot.iterate(f"{filepath}:Events", filter_name="/^ch[0-9]+$/", library="np", step_size="100 MB"):
        
        chunk_results = []
        for channel_name, waveforms in data_dict.items():
            
            
            if waveforms.dtype == 'O': 
                waveforms = np.vstack(waveforms)

            # Baseline calculations
            baseline_window = waveforms[:, baseline_start:baseline_end] 
            baselines = np.mean(baseline_window, axis=1)
            std_devs = np.std(baseline_window, axis=1)
                
            corrected_waveforms = waveforms - baselines[:, None]
            corrected_peak_window = corrected_waveforms[:, peak_start:peak_end]
            
            # Find negative peaks specifically (minimum values)
            peaks = np.argmin(corrected_peak_window, axis=1) + peak_start
            peak_values = corrected_waveforms[np.arange(len(waveforms)), peaks]
            
            # 3-Sigma Cut toggle
            if apply_3sigma_cut:
                valid_mask = peak_values < (-3 * std_devs)
            else:
                valid_mask = np.ones(len(peak_values), dtype=bool)
            
            # ---------------------------------------------------------
            # OPTIMIZATION: Filter arrays BEFORE doing heavy Numpy math
            # ---------------------------------------------------------
            valid_waveforms = corrected_waveforms[valid_mask]
            valid_peaks = peaks[valid_mask]
            valid_baselines = baselines[valid_mask]
            valid_std_devs = std_devs[valid_mask]
            
            if len(valid_waveforms) == 0:
                continue 

            # Pure NumPy zero-crossing and integration (applied ONLY to valid signals)
            num_samples = valid_waveforms.shape[1]
            indices = np.arange(num_samples)
            
            above_baseline = (valid_waveforms >= 0)

            left_valid = above_baseline & (indices < valid_peaks[:, None])
            left_crossings = np.max(np.where(left_valid, indices, 0), axis=1)

            right_valid = above_baseline & (indices > valid_peaks[:, None])
            right_crossings = np.min(np.where(right_valid, indices, num_samples - 1), axis=1)

            integration_mask = (indices >= left_crossings[:, None]) & (indices <= right_crossings[:, None])
            integrals = np.sum(valid_waveforms * integration_mask, axis=1)

            # Build DataFrame only for valid signals
            chunk_df = pd.DataFrame({
                "channel": channel_name,
                "baseline": valid_baselines,
                "std_dev": valid_std_devs,
                "peak_integral": integrals
            })
            chunk_results.append(chunk_df)

        # 2. STREAM DIRECTLY TO PARQUET
        if chunk_results:
            combined_chunk = pd.concat(chunk_results, ignore_index=True)
            table = py.Table.from_pandas(combined_chunk)
            
            if pq_writer is None:
                pq_writer = pq.ParquetWriter(final_parquet_path, table.schema)
            pq_writer.write_table(table)

    # 3. FINALIZE AND RETURN
    if pq_writer is not None:
        pq_writer.close()
        print(f"Data successfully saved to {final_parquet_path}")
    else:
        print("No valid events found to save.")














def analyse_root_data_type_b(filepath, parquet_folder=None, suffix=None, apply_3sigma_cut=True):
    """
    Processes File Type B: 
    Data stored in "Data_R;X" trees with "Samples" and "Channel" branches.
    Finds POSITIVE peaks.
    """
    # Determine integration windows based on device
    baseline_start, baseline_end = baseline_start_4, baseline_end_4
    peak_start, peak_end = peak_start_4, peak_end_4
    

    # Setup output path
    filepath_str = str(filepath)
    original_name = Path(filepath).stem
    if suffix is not None:
        original_name += f"_{suffix}"
        
    if parquet_folder is None:
        parquet_folder = Path(filepath).parent
        
    final_parquet_path = Path(parquet_folder) / f"{original_name}.parquet"
    final_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    
    pq_writer = None

    # 1. IDENTIFY ALL RELEVANT TREES
    with uproot.open(filepath_str) as f:
        keys = f.keys()
        data_r_keys = [k for k in keys if k.startswith("Data_R")]

    if not data_r_keys:
        print(f"No 'Data_R' trees found in {filepath_str}")
        return None

    # 2. PROCESS IN CHUNKS ACROSS ALL TREES
    for tree_key in data_r_keys:
        for chunk in uproot.iterate({filepath_str: tree_key}, filter_name=["Samples", "Channel"], library="np", step_size="100 MB"):
            
            samples = chunk["Samples"]
            channels = chunk["Channel"]
            unique_channels = np.unique(channels)
            
            chunk_results = []
            
            for ch in unique_channels:
                channel_name = f"ch{int(ch):02d}"
                mask = (channels == ch)
                waveforms = samples[mask]
                
                # --- EXACT COMPUTATION LOGIC START ---
                if waveforms.dtype == 'O': 
                    waveforms = np.vstack(waveforms)
                    
                # Array Safety Checks
                if len(waveforms) == 0:
                    continue
                    
                num_samples = waveforms.shape[1]
                if peak_start >= num_samples or baseline_start >= num_samples:
                    continue
                if peak_start >= peak_end:
                    raise ValueError(f"Invalid peak window: start ({peak_start}) must be less than end ({peak_end}).")

                # Baseline calculations
                baseline_window = waveforms[:, baseline_start:baseline_end] 
                baselines = np.mean(baseline_window, axis=1)
                std_devs = np.std(baseline_window, axis=1)
                    
                corrected_waveforms = waveforms - baselines[:, None]
                corrected_peak_window = corrected_waveforms[:, peak_start:peak_end]
                
                # --- CHANGED: Find POSITIVE peaks specifically (maximum values) ---
                peaks = np.argmax(corrected_peak_window, axis=1) + peak_start
                peak_values = corrected_waveforms[np.arange(len(waveforms)), peaks]
                
                # --- CHANGED: 3-Sigma Cut toggle (Positive Side) ---
                if apply_3sigma_cut:
                    valid_mask = peak_values > (3 * std_devs)
                else:
                    valid_mask = np.ones(len(peak_values), dtype=bool)
                
                # OPTIMIZATION: Filter arrays BEFORE doing heavy Numpy math
                valid_waveforms = corrected_waveforms[valid_mask]
                valid_peaks = peaks[valid_mask]
                valid_baselines = baselines[valid_mask]
                valid_std_devs = std_devs[valid_mask]
                
                if len(valid_waveforms) == 0:
                    continue 

                # Pure NumPy zero-crossing and integration (applied ONLY to valid signals)
                num_samples = valid_waveforms.shape[1]
                indices = np.arange(num_samples)
                
                # --- CHANGED: Look for crossings below baseline for positive peaks ---
                below_baseline = (valid_waveforms <= 0)

                left_valid = below_baseline & (indices < valid_peaks[:, None])
                left_crossings = np.max(np.where(left_valid, indices, 0), axis=1)

                right_valid = below_baseline & (indices > valid_peaks[:, None])
                right_crossings = np.min(np.where(right_valid, indices, num_samples - 1), axis=1)

                integration_mask = (indices >= left_crossings[:, None]) & (indices <= right_crossings[:, None])
                integrals = -1 *np.sum(valid_waveforms * integration_mask, axis=1)

                # Build DataFrame only for valid signals
                chunk_df = pd.DataFrame({
                    "channel": channel_name,
                    "baseline": valid_baselines,
                    "std_dev": valid_std_devs,
                    "peak_integral": integrals
                })
                chunk_results.append(chunk_df)
                # --- EXACT COMPUTATION LOGIC END ---

            # 3. STREAM DIRECTLY TO PARQUET
            if chunk_results:
                combined_chunk = pd.concat(chunk_results, ignore_index=True)
                table = py.Table.from_pandas(combined_chunk)
                
                if pq_writer is None:
                    pq_writer = pq.ParquetWriter(final_parquet_path, table.schema)
                pq_writer.write_table(table)

    # 4. FINALIZE AND RETURN
    if pq_writer is not None:
        pq_writer.close()
        print(f"Data successfully saved to {final_parquet_path}")
    else:
        print("No valid events found to save.")


   








def analyse_dark_counts(filepath, cut_sigma=3.0, suffix=None, parquet_folder="."):
    analyzed_data = [] 

    # 1. PROCESS IN CHUNKS
    for data_dict in uproot.iterate(f"{filepath}:Events", filter_name="/^ch[0-9]+$/", library="np", step_size="100 MB"):
        
        chunk_results = []
        for channel_name, waveforms in data_dict.items():
            print(f"Processing dark counts for {channel_name}...")
            
            if waveforms.dtype == 'O': 
                waveforms = np.vstack(waveforms)

            # --- AUTOMATIC BASELINE DETECTION ---
            # Use the entire waveform. The median ignores the short dark count pulses.
            baselines = np.median(waveforms, axis=1)
            
            # Median Absolute Deviation (MAD) calculates the noise spread while ignoring peaks.
            mad = np.median(np.abs(waveforms - baselines[:, None]), axis=1)
            
            # Multiply by 1.4826 to convert MAD into an equivalent standard deviation.
            # We use np.maximum to ensure std_devs is never exactly 0 (prevents math errors on flatlined data).
            std_devs = np.maximum(mad * 1.4826, 1e-6)
                
            corrected_waveforms = waveforms - baselines[:, None]
            
            # Fast Pre-filter: Check which waveforms have AT LEAST ONE point below the threshold.
            thresholds = -cut_sigma * std_devs
            has_dark_count = np.any(corrected_waveforms < thresholds[:, None], axis=1)
            
            # Lists to store the individual dark counts
            chan_list = []
            bl_list = []
            std_list = []
            integral_list = []
            
            # Extract indices of waveforms that passed the quick filter
            candidate_indices = np.where(has_dark_count)[0]
            
            for idx in candidate_indices:
                cw = corrected_waveforms[idx]
                std = std_devs[idx]
                bl = baselines[idx]
                thresh = thresholds[idx]
                
                # Find everywhere the waveform is below the baseline
                below_zero = cw < 0
                
                # Identify the start and end indices of these continuous below-zero regions
                edges = np.diff(below_zero.astype(int))
                starts = np.where(edges == 1)[0] + 1
                ends = np.where(edges == -1)[0] + 1
                
                # Handle edge cases
                if below_zero[0]:
                    starts = np.insert(starts, 0, 0)
                if below_zero[-1]:
                    ends = np.append(ends, len(cw))
                    
                # Evaluate each continuous negative region independently
                for s, e in zip(starts, ends):
                    segment = cw[s:e]
                    
                    # If this specific peak crosses the sigma threshold, integrate it
                    if np.min(segment) < thresh:
                        chan_list.append(channel_name)
                        bl_list.append(bl)
                        std_list.append(std)
                        integral_list.append(np.sum(segment))
            
            # If we found any valid dark counts in this chunk, save them
            if len(chan_list) > 0:
                chunk_results.append(pd.DataFrame({
                    "channel": chan_list,
                    "baseline": bl_list,
                    "std_dev": std_list,
                    "peak_integral": integral_list
                }))

            # 3. MANUALLY FREE MEMORY
            del waveforms, corrected_waveforms, has_dark_count, thresholds
            gc.collect()

        if chunk_results:
            analyzed_data.append(pd.concat(chunk_results, ignore_index=True))
        
    if not analyzed_data:
        print("No dark counts found in the entire file.")
        return None, None
        
    master_df = pd.concat(analyzed_data, ignore_index=True)
    
    original_name = Path(filepath).parent.name
    if suffix is not None:
        original_name += f"_{suffix}"
        
    final_parquet_path = Path(parquet_folder) / f"{original_name}.parquet"
    final_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the combined DataFrame
    master_df.to_parquet(final_parquet_path, index=False)
    print(f"Data successfully saved to {final_parquet_path}")
        
    return master_df, final_parquet_path









# --------------------------------------------------------- 

def fit_model_physics(x, *params): 
    """ 
    Physics-linked model for SiPM/PMT multi-photon spectra. 
    params[0]: A_bg       (Exponential background amplitude) 
    params[1]: lambda_bg  (Exponential background decay rate) 
    params[2]: mu_0       (Position of the 0-photon pedestal) 
    params[3]: sigma_0    (Width of the 0-photon pedestal) 
    params[4]: gain       (Distance between adjacent photon peaks) 
    params[5]: sigma_1    (Additional width contribution per photon) 
    params[6]: C          (Constant baseline offset)
    params[7:]: A_0, A_1, A_2... (Amplitudes for each photon peak) 
    """ 
    A_bg = params[0] 
    lambda_bg = params[1] 
    mu_0 = params[2] 
    sigma_0 = params[3] 
    gain = params[4] 
    sigma_1 = params[5] 
    C = params[6]
    amplitudes = params[7:] 
    
    # Calculate Background (Shifted Exponential + Constant Baseline)
    y = np.where(x >= mu_0, A_bg * np.exp(-lambda_bg * (x - mu_0)), 0.0) + C
    
    # Add Physics-Linked Gaussians 
    for n, A in enumerate(amplitudes): 
        mu_n = mu_0 + n * gain 
        sigma_n = np.sqrt(sigma_0**2 + n * sigma_1**2) 
        y += A * np.exp(-0.5 * ((x - mu_n) / sigma_n)**2) 
        
    return y


def fit_pandas_data(raw_data, channel_name="Data", bins=500, num_peaks_to_fit=18, 
                    confident_peak_limit=7, manual_mu0=None, manual_gain=None, ax=None,
                    max_reduced_chi2=None): 
    
    raw_data = np.array(raw_data.dropna(), dtype=float) 
    print(f"Processing {len(raw_data)} data points for {channel_name}...") 

    counts, bin_edges = np.histogram(raw_data, bins=bins) 
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 
    dx = bin_centers[1] - bin_centers[0] 
    
    fit_successful = False
    popt, pcov = None, None
    peak_params = [] 
    
    reduced_chi_square = np.nan
    chi_square = np.nan
    ndf = 0

    if manual_mu0 is not None and manual_gain is not None: 
        print("Using manual physics overrides for Pedestal and Gain.") 
        mu_0_guess = manual_mu0 
        gain_guess = manual_gain 
    else: 
        print("Auto-detecting peaks...") 
        smoothed_counts = gaussian_filter1d(counts, sigma=1) 
        peaks, _ = find_peaks( 
            smoothed_counts,  
            prominence=np.max(smoothed_counts) * 0.01,  
            distance=4 
        ) 
        
        global_max_bin = np.argmax(counts) 
        if not any(abs(p - global_max_bin) <= 5 for p in peaks): 
            peaks = np.append(peaks, global_max_bin) 
            peaks = np.sort(peaks) 

        if len(peaks) == 0: 
            print("Failed to find any peaks. Skipping fit and plotting raw data.")
        else:
            idx_0 = peaks[0] 
            mu_0_guess = bin_centers[idx_0] 
            
            if len(peaks) > 1: 
                gain_guess = bin_centers[peaks[1]] - bin_centers[peaks[0]] 
            else: 
                gain_guess = (bin_centers[-1] - bin_centers[0]) / (num_peaks_to_fit / 2) 

    if (manual_mu0 is not None) or (len(peaks) > 0):
        sigma_0_guess = max(dx, 1e-9) 
        sigma_1_guess = sigma_0_guess * 0.5  

        p0 = [] 
        lower_bounds = [] 
        upper_bounds = [] 

        A_bg_guess = np.max(counts) * 0.1 
        lambda_bg_guess = 1.0 / max(np.abs(np.mean(raw_data)), 1.0) 
        p0.extend([A_bg_guess, lambda_bg_guess]) 
        lower_bounds.extend([0, 0]) 
        upper_bounds.extend([np.max(counts) * 0.5, 5000]) 

        p0.extend([mu_0_guess, sigma_0_guess, gain_guess, sigma_1_guess]) 
        lower_bounds.extend([bin_centers[0] - dx*10, 1e-12, dx, 1e-12]) 
        upper_bounds.extend([bin_centers[-1], gain_guess * 1.2, gain_guess * 5, gain_guess * 1.2]) 

        # Add Baseline Constant Bounds (Index 6)
        C_guess = 1.0
        p0.append(C_guess)
        lower_bounds.append(0.0)
        upper_bounds.append(np.max(counts) * 0.1)

        for n in range(num_peaks_to_fit): 
            expected_mu = mu_0_guess + (n * gain_guess) 
            if bin_centers[0] <= expected_mu <= bin_centers[-1]: 
                closest_bin_idx = np.abs(bin_centers - expected_mu).argmin() 
                A_guess = counts[closest_bin_idx] 
            else: 
                A_guess = 1e-9  
                
            p0.append(A_guess) 
            lower_bounds.append(0.0) 
            upper_bounds.append(np.inf) 

        p0 = np.clip(np.array(p0, dtype=float), np.array(lower_bounds, dtype=float) + 1e-10, np.array(upper_bounds, dtype=float) - 1e-10) 

        # Poisson error + 2% systematic error floor
        y_err = np.maximum(np.sqrt(counts), 1.0) + (0.02 * counts)

        try: 
            popt_temp, pcov_temp = curve_fit( 
                fit_model_physics,  
                bin_centers,  
                counts,  
                p0=p0, 
                sigma=y_err,          
                absolute_sigma=True,  
                bounds=(lower_bounds, upper_bounds), 
                maxfev=25000  
            ) 
            
            expected_counts = fit_model_physics(bin_centers, *popt_temp)
            
            # Use the exact same weighting used in the optimizer for chi-square
            variances = y_err ** 2
            
            chi_square = np.sum(((counts - expected_counts) ** 2) / variances)
            ndf = len(counts) - len(popt_temp)
            reduced_chi_square = chi_square / ndf if ndf > 0 else np.nan
            
            if max_reduced_chi2 is not None and reduced_chi_square > max_reduced_chi2:
                print(f"Fit rejected: Reduced \u03C7\u00B2 ({reduced_chi_square:.2f}) > threshold ({max_reduced_chi2}).")
            else:
                print("Physics Curve fitting converged successfully!") 
                fit_successful = True
                popt = popt_temp
                pcov = pcov_temp
                
                # Unpack the 7 physical parameters
                A_bg, lambda_bg, mu_0, sigma_0, gain, sigma_1, C = popt[0:7] 
                amplitudes = popt[7:] 
                
                for n in range(len(amplitudes)):
                    mu_n = mu_0 + n * gain
                    sigma_n = np.sqrt(sigma_0**2 + n * sigma_1**2)
                    peak_params.append([mu_n, sigma_n])
            
        except Exception as e: 
            print(f"Optimal parameters not found: {e}") 

    if ax is not None:
        ax.hist(raw_data, bins=bins, alpha=0.5, color='blue', label='Binned Data') 
        
        if fit_successful:
            x_fit = np.linspace(bin_centers[0], bin_centers[-1], 1000) 
            ax.plot(x_fit, fit_model_physics(x_fit, *popt), color='red', lw=2, label='Total Fit') 
            
            print(f"\n--- Physical Fit Results for {channel_name} ---") 
            for n, (mu_n, sigma_n) in enumerate(peak_params[:4]):
                print(f"  {n}-Photon Peak: \u03BC = {mu_n:.6e}, \u03C3 = {sigma_n:.6e}")
            print("-" * 50) 
            print(f"\u03C7\u00B2 / NDF:           {chi_square:.2f} / {ndf} = {reduced_chi_square:.3f}")
            print("-" * 28) 
            
            tail_sum = np.zeros_like(x_fit) 
            for n, (mu_n, sigma_n) in enumerate(peak_params): 
                A = popt[7 + n]
                single_gauss = A * np.exp(-0.5 * ((x_fit - mu_n) / sigma_n)**2) 
                
                if n <= confident_peak_limit: 
                    if np.max(single_gauss) > (np.max(counts) * 0.005): 
                        ax.plot(x_fit, single_gauss, '--', label=f'{n}-Photon') 
                else: 
                    tail_sum += single_gauss 

            if np.max(tail_sum) > (np.max(counts) * 0.005): 
                ax.plot(x_fit, tail_sum, color='gray', linestyle='-.', lw=1.5, label='High-Photon Tail') 

            ax.set_title(f'{channel_name}, with \u03C7\u00B2/NDF = {reduced_chi_square:.3f}') 
        else:
            if not np.isnan(reduced_chi_square) and max_reduced_chi2 is not None and reduced_chi_square > max_reduced_chi2:
                ax.set_title(f'{channel_name} (REJECTED: \u03C7\u00B2/NDF > {max_reduced_chi2})') 
            else:
                ax.set_title(f'{channel_name} (FIT FAILED)') 

        ax.set_xlabel('ADC Counts') 
        ax.set_ylabel('Frequency') 
        ax.grid(alpha=0.3) 

    return popt, pcov, np.array(peak_params)



'''
def fit_all_channels_in_df(parquet, column_to_fit="peak_integral", output_folder=None, file_id=None, **kwargs):
    """
    Iterates through every unique channel in the dataframe, extracts the data,
    and runs the physics fit on it.
    
    If output_folder is provided, it extracts the name of the parquet file, creates a 
    subfolder with that name inside the output_folder (if it doesn't exist), 
    and saves the histograms there.
    
    Returns a Pandas DataFrame of the optimized parameters for each channel.
    """
    fit_results = {}
    df = pd.read_parquet(parquet)
    
    # Find all unique channels in the DataFrame
    channels = df['channel'].unique()
    num_plots = len(channels)
    
    # ==========================================
    # Setup the single canvas grid if plotting
    # ==========================================
    axes = None
    fig = None
    if output_folder is not None and num_plots > 0:
        cols = min(10, num_plots)
        rows = (num_plots - 1) // 10 + 1  
        
        fig, axes = plt.subplots(nrows=rows, ncols=cols, 
                                 figsize=(4 * cols, 3.5 * rows), 
                                 squeeze=False)
        axes = axes.flatten()

    # ==========================================
    # Iterate and Fit
    # ==========================================
    for idx, ch in enumerate(channels):
        print(f"\n=====================================")
        print(f" STARTING ANALYSIS FOR: {ch}")
        print(f"=====================================")
        
        channel_data = df[df['channel'] == ch][column_to_fit]
        current_ax = axes[idx] if axes is not None else None
        
        try:
            popt, pcov, peak_params = fit_pandas_data(
                channel_data, 
                channel_name=ch, 
                ax=current_ax, 
                **kwargs
            )
        except Exception as e:
            print(f"WARNING: Fit failed for channel {ch}. Error: {e}")
            continue # Skip to the next channel

        if popt is not None:
            channel_dict = {
                "base_mu_0": popt[2],       
                "gain": popt[4],            
                "base_sigma_0": popt[3],    
                "base_sigma_1": popt[5],    
                "raw_popt": popt,
                "raw_pcov": pcov
            }
            
            for n, (mu, sigma) in enumerate(peak_params):
                channel_dict[f"mu_{n}"] = mu
                channel_dict[f"sigma_{n}"] = sigma
                
            fit_results[ch] = channel_dict
            
    # ==========================================
    # Finalize and Save the Canvas
    # ==========================================
    if output_folder is not None and num_plots > 0:
        # Hide any unused axes
        for idx in range(num_plots, len(axes)):
            axes[idx].set_visible(False)
            
        plt.tight_layout()
        
        # Extract the base name of the parquet file (e.g., "run_001.parquet" -> "run_001")
        file_basename = os.path.splitext(os.path.basename(parquet))[0]
        
        # Create the nested directory path
        target_dir = os.path.join(output_folder, file_basename)
        
        # Create directory (ignores if it exists already)
        os.makedirs(target_dir, exist_ok=True)
        
        save_file = os.path.join(target_dir, "all_channels_canvas.pdf")
        plt.savefig(save_file)
        print(f"\nAll plots successfully saved to {save_file}")
        
        plt.close('all')

    # ==========================================
    # Build DataFrame
    # ==========================================
    results_df = pd.DataFrame.from_dict(fit_results, orient='index')
    
    if not results_df.empty:
        results_df.index.name = 'channel'
        results_df = results_df.reset_index()
        
        if file_id is not None:
            results_df.insert(0, 'file_id', file_id)

   
        
    
    gc.collect()
'''


import os
import gc
import pandas as pd
import matplotlib.pyplot as plt

def fit_all_channels_in_df(
    parquet, 
    column_to_fit="peak_integral", 
    output_folder=None, 
    file_id=None,
    log=False, 
    lower_percentile=2.0, 
    upper_percentile=99.9,
    **kwargs
):
    """
    Iterates through every unique channel in the dataframe, extracts the data,
    and runs the physics fit on it using fit_single_channel_data.
    
    If output_folder is provided, it extracts the name of the parquet file, creates a 
    subfolder with that name inside the output_folder (if it doesn't exist), 
    and saves the histograms there.
    
    Returns a Pandas DataFrame of the optimized parameters for each channel.
    """
    fit_results = {}
    df = pd.read_parquet(parquet)
    metadata=parse_folder_metadata(parquet)
    
    # Find all unique channels in the DataFrame
    channels = df['channel'].unique()
    num_plots = len(channels)
    
    # ==========================================
    # Setup the single canvas grid if plotting
    # ==========================================
    axes = None
    fig = None
    if output_folder is not None and num_plots > 0:
        cols = min(10, num_plots)
        rows = (num_plots - 1) // 10 + 1  
        
        fig, axes = plt.subplots(nrows=rows, ncols=cols, 
                                 figsize=(4 * cols, 3.5 * rows), 
                                 squeeze=False)
        axes = axes.flatten()

    # ==========================================
    # Iterate and Fit
    # ==========================================
    for idx, ch in enumerate(channels):
        print(f"\n=====================================")
        print(f" STARTING ANALYSIS FOR: {ch}")
        print(f"=====================================")
        
        channel_data = df[df['channel'] == ch][column_to_fit]
        current_ax = axes[idx] if axes is not None else None
        
        try:
            # 1. Run the new fit function with explicit percentiles
            res = fit_single_channel_data(
                raw_data=channel_data, 
                channel=ch, 
                lower_percentile=lower_percentile,
                upper_percentile=upper_percentile,
                **kwargs
            )
            
            popt = res.get('popt')
            pcov = res.get('pcov')
            peak_params = res.get('peak_params', [])
            
            # 2. Plot onto the current axis using the dedicated plotting function
            if current_ax is not None:
                plot_peak_area_histogram(
                    ax=current_ax,
                    fit_results=res,
                    color='tab:blue', # Using the standard matplotlib blue which is slightly softer
                    metadata=metadata, 
                    target_channel=ch,
                    log_scale=log 
                )
                
                # Re-add the missing vertical lines and mu text box
                if res.get("fit_successful") and len(peak_params) > 0:
                    first_mus = peak_params[:5, 0]
                    for mu in first_mus:
                        current_ax.axvline(x=mu, color='black', linestyle=':', alpha=0.7)
                    
                    mu_text = "\n".join([f"μ{idx_mu}: {mu:.0f}" for idx_mu, mu in enumerate(first_mus)])
                    current_ax.text(0.98, 0.80, mu_text, transform=current_ax.transAxes, va='top', ha='right', 
                                    fontsize=9, bbox=dict(facecolor='white', alpha=0.9, edgecolor='lightgray', pad=4))
                
        except Exception as e:
            print(f"WARNING: Fit failed for channel {ch}. Error: {e}")
            continue # Skip to the next channel

        # 3. Extract the physical parameters for the summary DataFrame
        if popt is not None and res.get('fit_successful', False):
            channel_dict = {
                "base_mu_0": popt[2],       
                "gain": popt[4],            
                "base_sigma_0": popt[3],    
                "base_sigma_1": popt[5],    
                "raw_popt": popt,
                "raw_pcov": pcov
            }
            
            for n, (mu, sigma) in enumerate(peak_params):
                channel_dict[f"mu_{n}"] = mu
                channel_dict[f"sigma_{n}"] = sigma
                
            fit_results[ch] = channel_dict
            
    # ==========================================
    # Finalize and Save the Canvas
    # ==========================================
    if output_folder is not None and num_plots > 0:
        # Hide any unused axes
        for idx in range(num_plots, len(axes)):
            axes[idx].set_visible(False)
            
        plt.tight_layout()
        
        # Extract the base name of the parquet file
        file_basename = os.path.splitext(os.path.basename(parquet))[0]
        
        # Create the nested directory path
        target_dir = os.path.join(output_folder, file_basename)
        os.makedirs(target_dir, exist_ok=True)
        
        save_file = os.path.join(target_dir, "all_channels_canvas.pdf")
        plt.savefig(save_file)
        print(f"\nAll plots successfully saved to {save_file}")
        
        plt.close('all')

    # ==========================================
    # Build DataFrame
    # ==========================================
    results_df = pd.DataFrame.from_dict(fit_results, orient='index')
    
    if not results_df.empty:
        results_df.index.name = 'channel'
        results_df = results_df.reset_index()
        
        if file_id is not None:
            results_df.insert(0, 'file_id', file_id)

    gc.collect()
    
    return results_df





#input a root file with data from several channels
#plots several randomly picket events for each channel
def plot_sample_signals_from_root(filepath, num_samples=2, device=None, plot_folder="."):
    """
    Plots sample signals from a DAQ ROOT file for all available channels 
    on a single grid canvas, shading the exact integration window.

    Parameters:
    filepath (str): Path to the ROOT file.
    num_samples (int): Number of sample plots to generate per channel.
    device (str): Device type for setting hardcoded window boundaries.
    plot_folder (str): Directory to save the output plots.
    """

    # 1. Open the ROOT file and extract the data
    with uproot.open(filepath) as f:
        tree = f['Events']
        # Read branches matching the channel regex into a dictionary of numpy arrays
        data_dict = tree.arrays(filter_name="/^ch[0-9]+$/", library="np")

    if not data_dict:
        print("Error: No branches matching '/^ch[0-9]+$/' were found.")
        return

    # Extract the metadata from the root file and use it to label the plots
    # (Assuming parse_folder_metadata is defined elsewhere in your code)
    metadata = parse_folder_metadata(filepath)

    if metadata.get("is_dark_count", False):
        conditions = [metadata['temp_c'], 0, metadata['bias_voltage']]
    else:
        conditions = [metadata['temp_c'], metadata['led_voltage'], metadata['bias_voltage']]

    # 2. Pre-process and collect all valid signals to plot
    plot_tasks = []
    for channel_name in data_dict.keys():
        print(f"Sampling {channel_name}...")
        samples_matrix = data_dict[channel_name]
        
        total_rows = len(samples_matrix)
        actual_num_samples = min(num_samples, total_rows)
        
        # Pick random row indices without replacement for the current channel
        random_indices = np.random.choice(total_rows, size=actual_num_samples, replace=False)
        
        for row_idx in random_indices:
            signal = np.asarray(samples_matrix[row_idx], dtype=float)
            valid_mask = ~np.isnan(signal)
            sig = signal[valid_mask]
            
            # Skip if signal is too short
            if len(sig) < 50:
                continue
                
            plot_tasks.append((channel_name, row_idx, sig))

    total_plots = len(plot_tasks)
    if total_plots == 0:
        print("No valid signals found to plot.")
        return

    # 3. Setup the subplot grid (max 10 columns)
    max_cols = 10
    ncols = min(max_cols, total_plots)
    nrows = int(np.ceil(total_plots / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 6, nrows * 4.5), squeeze=False)
    axes = axes.flatten()

    # 4. Iterate over collected tasks and plot them on their respective axes
    for i, (channel_name, row_idx, sig) in enumerate(plot_tasks):
        ax = axes[i]
        t = np.arange(len(sig))

        # Calculate baseline and sigma (using MEDIAN to match function 1)
        if device is not None:
            bkg_start = baseline_start_4
            bkg_end = baseline_end_4
            start_indx = max(0, peak_start_4)
            end_indx = min(len(t), peak_end_4)
        else:
            bkg_start = baseline_start_64
            bkg_end = baseline_end_64
            start_indx = max(0, peak_start_64)
            end_indx = min(len(t), peak_end_64)

        bkg_sig = sig[bkg_start:bkg_end]
        baseline = np.median(bkg_sig)
        sigma = np.std(bkg_sig, ddof=1) if len(bkg_sig) > 1 else 0.0

        # Plotting onto the specific axis
        ax.plot(t, sig, label=f'Signal ({channel_name})', color='blue')
        ax.axhline(y=baseline, color='green', linestyle='--', label='Baseline')
        ax.axhline(y=baseline + sigma, color='orange', linestyle='--', label='+/- 1 Sigma')
        ax.axhline(y=baseline - sigma, color='orange', linestyle='--')
        
        # Keep the visual markers for the hardcoded window boundaries
        if 0 <= start_indx < len(t):
            ax.axvline(x=t[start_indx], color='black', linestyle='--', label='Hardcoded window')
        if 0 <= end_indx - 1 < len(t):
            ax.axvline(x=t[end_indx - 1], color='black', linestyle='--')

        # --- STRICT NEGATIVE PEAK INTEGRATION LOGIC (Matches analyse_root_data) ---
        sig_window_initial = sig[start_indx:end_indx]
        
        if len(sig_window_initial) > 0:
            # 1. Find the MINIMUM peak inside the hardcoded region (Negative signals)
            peak_idx = start_indx + np.argmin(sig_window_initial)
                
            # 2. Walk LEFT from the peak until the signal goes above the baseline
            left_idx = peak_idx
            while left_idx > 0 and sig[left_idx] < baseline:
                left_idx -= 1
                    
            # 3. Walk RIGHT from the peak until the signal goes above the baseline
            right_idx = peak_idx
            while right_idx < len(sig) - 1 and sig[right_idx] < baseline:
                right_idx += 1

            # 4. Extract exactly the window defined by the baseline crossings
            # We add +1 to right_idx to include the boundary point for plotting
            t_window = t[left_idx:right_idx + 1]
            sig_window = sig[left_idx:right_idx + 1]

            # 5. Color the integrated area
            if len(t_window) > 0:
                ax.fill_between(
                    t_window, 
                    sig_window, 
                    baseline, 
                    color='purple', 
                    alpha=0.3, 
                    label='Integrated Peak'
                )
        # ----------------------------------------
       
        # Titles and labels
        if metadata.get("is_dark_count", False):
            ax.set_title(
                f'Sample Signal {i+1} (Row: {row_idx}, Ch: {channel_name})\n'
                f'{float(conditions[0]):.1f} °C / {float(conditions[0])+273.15:.1f} K\n'
                f'No LED, Dark count measurement | SiPM: {float(conditions[2]):.1f} V',
                fontsize=10
            )
        else:    
            ax.set_title(
                f'Sample Signal {i+1} (Row: {row_idx}, Ch: {channel_name})\n'
                f'{float(conditions[0]):.1f} °C / {float(conditions[0])+273.15:.1f} K\n'
                f'LED: {float(conditions[1]):.1f} V | SiPM: {float(conditions[2]):.1f} V',
                fontsize=10
            )
        
        ax.set_xlabel('Samples')
        ax.set_ylabel('Signal (ADC)')
        ax.legend(fontsize=8)
        ax.grid(True)

    # 5. Clean up any unused subplots
    for j in range(total_plots, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    # 6. Display/Save the combined figure
    original_folder_name = Path(filepath).parent.name
    save_dir = Path(plot_folder) / original_folder_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    plot_filename = "combined_sample_signals.pdf"
    full_save_path = save_dir / plot_filename
    
    plt.savefig(full_save_path, dpi=200)
    print(f"Combined plot saved to: {full_save_path}")

    plt.show()

    return save_dir















def plot_average_signals_from_root(filepath, plot_folder, baseline_window=(baseline_start_64,baseline_end_64), peak_window=(peak_start_64,peak_end_64)):
    """
    Computes and plots the average signal pulse from a DAQ ROOT file for all 
    available channels, displaying each channel in a separate subplot.
    The integration (shaded) window is set exactly to the peak search window.

    Parameters:
    filepath (str): Path to the ROOT file.
    plot_folder (str): Directory to save the generated plot.
    baseline_window (tuple): (start, end) indices for baseline calculation.
    peak_window (tuple): (start, end) indices for peak search and shaded integration window.
    """
    # Ensure the output directory exists
    os.makedirs(plot_folder, exist_ok=True)

    # 1. Open the ROOT file and extract the data
    with uproot.open(filepath) as f:
        tree = f['Events']
        data_dict = tree.arrays(filter_name="/^ch[0-9]+$/", library="np")

    if not data_dict:
        print("Error: No channels found matching '/^ch[0-9]+$/'.")
        return

    n_channels = len(data_dict)
    
    # 2. Determine grid size for subplots
    cols = math.ceil(math.sqrt(n_channels))
    rows = math.ceil(n_channels / cols)
    
    # Create the figure and axes grid
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    axes = axes.flatten()

    # 3. Process each channel and plot on its dedicated subplot
    for idx, (ch_name, events) in enumerate(data_dict.items()):
        ax = axes[idx]
        
        # Handle jagged arrays if event lengths vary
        waveforms = np.array(events.tolist()) if events.dtype == 'O' else np.atleast_2d(events)
        
        if waveforms.size == 0 or waveforms.ndim < 2:
            ax.set_title(f"{ch_name} (No valid data)")
            ax.axis('off')
            continue
            
        n_samples = waveforms.shape[1]
        
        # Baseline subtraction
        b_start, b_end = baseline_window
        baselines = np.mean(waveforms[:, b_start:b_end], axis=1, keepdims=True)
        waveforms_subtracted = waveforms - baselines
        
        # Compute average
        avg_waveform = np.mean(waveforms_subtracted, axis=0)
        
        # Define exact integration bounds based on peak_window
        int_start = max(0, peak_window[0])
        int_end = min(n_samples, peak_window[1])
        
        # Find peak strictly within that window
        if int_start < int_end:
            peak_idx = np.argmin(avg_waveform[int_start:int_end]) + int_start
        else:
            peak_idx = np.argmin(avg_waveform)
            
        time_axis = np.arange(n_samples)
        
        # Plotting on the specific subplot
        ax.plot(time_axis, avg_waveform, color='#1f77b4', lw=2)
        ax.axvspan(int_start, int_end, color='red', alpha=0.15)
        ax.plot(peak_idx, avg_waveform[peak_idx], 'ro', markersize=5)
        
        # Formatting for the subplot
        ax.set_title(f"{ch_name} (n={len(waveforms)})")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("ADC Counts")
        ax.grid(True, linestyle='--', alpha=0.7)

    # 4. Hide any unused subplots
    for i in range(n_channels, len(axes)):
        fig.delaxes(axes[i])

    # Final layout adjustments and save
    plt.tight_layout()
    plot_path = os.path.join(plot_folder, "average_signals_grid.pdf")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    gc.collect()
    
    print(f"Successfully processed {n_channels} channels. Grid plot saved to '{plot_path}'.")















'''
def plot_sample_signals_from_parquet(filepath, num_samples=2, device=None):
    """
    Plots sample signals from a DAQ Parquet file for all available channels 
    on a single grid canvas.

    Parameters:
    filepath (str): Path to the Parquet file.
    num_samples (int): Number of sample plots to generate per channel.
    device (str/None): Device identifier for baseline and peak window selection.
    """

    # 1. Open the Parquet file and extract the data
    try:
        df = pd.read_parquet(filepath)
    except Exception as e:
        print(f"Error opening Parquet file: {e}")
        return

    # Filter columns matching the channel regex and build the data dictionary
    channel_cols = [col for col in df.columns if re.match(r'^ch[0-9]+$', str(col))]
    
    data_dict = {}
    for col in channel_cols:
        # Convert the column (which should contain arrays/lists of signal data) to a numpy array
        data_dict[col] = df[col].to_numpy()

    if not data_dict:
        print("Error: No columns matching '^ch[0-9]+$' were found.")
        return

    # extract the metadata from the file path and use it to label the plots
    metadata = parse_folder_metadata(filepath)

    if metadata.get("is_dark_count", False):
        conditions = [metadata['temp_c'], 0, metadata['bias_voltage']]
    else:
        conditions = [metadata['temp_c'], metadata['led_voltage'], metadata['bias_voltage']]

    # 2. Pre-process and collect all valid signals to plot
    plot_tasks = []
    for channel_name in data_dict.keys():
        print(f"Sampling {channel_name}...")
        samples_matrix = data_dict[channel_name]
        
        total_rows = len(samples_matrix)
        actual_num_samples = min(num_samples, total_rows)
        
        # Pick random row indices without replacement for the current channel
        random_indices = np.random.choice(total_rows, size=actual_num_samples, replace=False)
        
        for row_idx in random_indices:
            signal = np.asarray(samples_matrix[row_idx], dtype=float)
            valid_mask = ~np.isnan(signal)
            sig = signal[valid_mask]
            
            # Skip if signal is too short
            if len(sig) < 50:
                continue
                
            plot_tasks.append((channel_name, row_idx, sig))

    total_plots = len(plot_tasks)
    if total_plots == 0:
        print("No valid signals found to plot.")
        return

    # 3. Setup the subplot grid (max 10 columns)
    max_cols = 10
    ncols = min(max_cols, total_plots)
    nrows = int(np.ceil(total_plots / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 6, nrows * 4.5), squeeze=False)
    axes = axes.flatten()

    # 4. Iterate over collected tasks and plot them on their respective axes
    for i, (channel_name, row_idx, sig) in enumerate(plot_tasks):
        ax = axes[i]
        t = np.arange(len(sig))

        # Calculate baseline and sigma
        if device is not None:
            bkg_start = baseline_start_4
            bkg_end = baseline_end_4
            start_indx = max(0, peak_start_4)
            end_indx = min(len(t), peak_end_4)
        else:
            bkg_start = baseline_start_64
            bkg_end = baseline_end_64
            start_indx = max(0, peak_start_64)
            end_indx = min(len(t), peak_end_64)

        bkg_sig = sig[bkg_start:bkg_end]
        baseline = np.mean(bkg_sig)
        sigma = np.std(bkg_sig, ddof=1) if len(bkg_sig) > 1 else 0.0

        # Plotting onto the specific axis
        ax.plot(t, sig, label=f'Signal ({channel_name})', color='blue')
        ax.axhline(y=baseline, color='green', linestyle='--', label='Baseline')
        ax.axhline(y=baseline + sigma, color='orange', linestyle='--', label='+/- 1 Sigma')
        ax.axhline(y=baseline - sigma, color='orange', linestyle='--')
        
        # Keep the visual markers for the hardcoded window boundaries
        if 0 <= start_indx < len(t):
            ax.axvline(x=t[start_indx], color='black', linestyle='--', label='Hardcoded window')
        if 0 <= end_indx - 1 < len(t):
            ax.axvline(x=t[end_indx - 1], color='black', linestyle='--')

        # --- STRICT SINGLE-PEAK SHADING LOGIC ---
        sig_window_initial = sig[start_indx:end_indx]
        
        if len(sig_window_initial) > 0:
            # 1. Find the maximum peak inside the hardcoded region
            max_val = np.max(sig_window_initial)
            min_val = np.min(sig_window_initial)
            
            # Determine if the main peak is positive or negative relative to baseline
            is_positive_peak = abs(max_val - baseline) >= abs(min_val - baseline)
            
            # Find the exact index of this peak in the full signal array
            if is_positive_peak:
                peak_idx = start_indx + np.argmax(sig_window_initial)
            else:
                peak_idx = start_indx + np.argmin(sig_window_initial)
                
            # 2. Walk LEFT from the peak until the first baseline crossing
            ext_start = peak_idx
            if is_positive_peak:
                while ext_start > 0 and sig[ext_start] > baseline:
                    ext_start -= 1
            else:
                while ext_start > 0 and sig[ext_start] < baseline:
                    ext_start -= 1
                    
            # 3. Walk RIGHT from the peak until the first baseline crossing
            ext_end = peak_idx
            if is_positive_peak:
                while ext_end < len(sig) - 1 and sig[ext_end] > baseline:
                    ext_end += 1
            else:
                while ext_end < len(sig) - 1 and sig[ext_end] < baseline:
                    ext_end += 1

            # 4. Extract just this single peak's strictly bounded window
            # We add +1 to ext_end to include the boundary point for interpolation
            t_window = t[ext_start:ext_end + 1]
            sig_window = sig[ext_start:ext_end + 1]

            # 5. Color the area (strictly one peak, zero chances of multiple baseline crosses)
            if is_positive_peak:
                fill_condition = (sig_window > baseline)
            else:
                fill_condition = (sig_window < baseline)

            if len(t_window) > 0:
                ax.fill_between(t_window, sig_window, baseline, 
                                where=fill_condition, 
                                color='purple', alpha=0.3, interpolate=True, label='Integrated Peak')
        # ----------------------------------------
        
        # Titles and labels
        if metadata.get("is_dark_count", False):
            ax.set_title(
                f'Sample Signal {i+1} (Row: {row_idx}, Ch: {channel_name})\n'
                f'{float(conditions[0]):.1f} °C / {float(conditions[0])+273.15:.1f} K\n'
                f'No LED, Dark count measurement | SiPM: {float(conditions[2]):.1f} V',
                fontsize=10
            )
        else:    
            ax.set_title(
                f'Sample Signal {i+1} (Row: {row_idx}, Ch: {channel_name})\n'
                f'{float(conditions[0]):.1f} °C / {float(conditions[0])+273.15:.1f} K\n'
                f'LED: {float(conditions[1]):.1f} V | SiPM: {float(conditions[2]):.1f} V',
                fontsize=10
            )
        ax.set_xlabel('Samples')
        ax.set_ylabel('Signal (ADC)')
        ax.legend(fontsize=8)
        ax.grid(True)

    # 5. Clean up any unused subplots
    for j in range(total_plots, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    # 6. Display/Save the combined figure
    original_folder_name = Path(filepath).parent.name
    
    # NOTE: Assuming `plot_folder` is defined globally or imported in your script
    save_dir = Path(plot_folder) / original_folder_name 
    save_dir.mkdir(parents=True, exist_ok=True)
    
    plot_filename = "combined_sample_signals.pdf"
    full_save_path = save_dir / plot_filename
    
    plt.savefig(full_save_path, dpi=200)
    print(f"Combined plot saved to: {full_save_path}")

    plt.show()

    return save_dir

'''


def plot_waveform_correlations(parquet, output_folder=None, channel=None, plot_type='hexbin'):
    """
    Plots baseline vs area, baseline vs sigma, and sigma vs area.
    Memory-optimized to prevent kernel crashes using pagination and selective loading.
    """
    # 1. OPTIMIZATION: Load only the columns we actually need
    columns_to_load = ['channel', 'baseline', 'peak_integral', 'std_dev']
    df = pd.read_parquet(parquet, columns=columns_to_load)
    
    if channel:
        channels_to_plot = [channel]
    else:
        channels_to_plot = sorted(df['channel'].unique())  

    num_channels = len(channels_to_plot)
    if num_channels == 0:
        print("No channels found to plot.")
        return

    # Configuration
    channels_per_page = 10  # Put 10 channels (30 plots) per page to prevent massive canvases
    
    plots = [
        {'x': 'baseline', 'y': 'peak_integral', 'xlab': 'Baseline', 'ylab': 'Area (Peak Integral)', 'title': 'Baseline vs Area'},
        {'x': 'baseline', 'y': 'std_dev', 'xlab': 'Baseline', 'ylab': 'Sigma (Std Dev)', 'title': 'Baseline vs Sigma'},
        {'x': 'std_dev', 'y': 'peak_integral', 'xlab': 'Sigma (Std Dev)', 'ylab': 'Area (Peak Integral)', 'title': 'Sigma vs Area'}
    ]

    # Set up saving paths
    pdf = None
    if output_folder is not None:
        file_basename = Path(parquet).stem
        save_dir = Path(output_folder) / file_basename
        save_dir.mkdir(parents=True, exist_ok=True)
        
        plot_filename = f"heatmaps_{channel}.pdf" if channel else "heatmaps_all_channels.pdf"
        full_save_path = save_dir / plot_filename
        pdf = PdfPages(full_save_path)
        
        # Turn off interactive mode to save memory
        plt.ioff()

    # 2. Iterate in chunks
    channel_chunks = [channels_to_plot[i:i + channels_per_page] 
                      for i in range(0, num_channels, channels_per_page)]

    for chunk_idx, chunk in enumerate(channel_chunks):
        chunk_size = len(chunk)
        
        # Create figure for the current page
        fig, axes = plt.subplots(nrows=chunk_size, ncols=3, 
                                 figsize=(18, 4.5 * chunk_size), 
                                 squeeze=False)
        
        # OPTIMIZATION: Manual spacing instead of tight_layout()
        plt.subplots_adjust(wspace=0.25, hspace=0.4, top=0.95, bottom=0.05)

        for row_idx, ch in enumerate(chunk):
            plot_df = df[df['channel'] == ch]
            
            for col_idx, p in enumerate(plots):
                ax = axes[row_idx, col_idx]
                
                if plot_type == 'hexbin':
                    hb = ax.hexbin(plot_df[p['x']], plot_df[p['y']], 
                                   gridsize=50, cmap='viridis', mincnt=1, 
                                   rasterized=True) # Rasterized keeps PDF size small
                    cb = fig.colorbar(hb, ax=ax)
                    cb.set_label('Counts')
                else:
                    # OPTIMIZATION: rasterized=True prevents massive vector files for scatter
                    ax.scatter(plot_df[p['x']], plot_df[p['y']], 
                               alpha=0.3, s=10, rasterized=True) 

                ax.set_xlabel(p['xlab'], fontsize=11)
                ax.set_ylabel(p['ylab'], fontsize=11)
                ax.set_title(f"[{ch}] {p['title']}", fontsize=14)
                ax.grid(True, linestyle='--', alpha=0.6)
                
            # Free memory for the sliced dataframe
            del plot_df

        # Save page and clear memory
        if pdf is not None:
            pdf.savefig(fig, dpi=100) # Lower DPI to save RAM; vector text stays crisp
            fig.clf()
            plt.close(fig)
            gc.collect()

    if pdf is not None:
        pdf.close()
        print(f"Heatmaps successfully saved to: {full_save_path}")
        plt.close('all')

    del df
    gc.collect()




def process_single_parquet_to_canvas(parquet_file, output_folder, max_reduced_chi2= 2,lower_percentile=2.0, upper_percentile=99.9, log=False):
    
       

    #fit the data for each channel obtained from the parquet file
    fit_all_channels_in_df(parquet_file ,output_folder = output_folder,lower_percentile=lower_percentile, 
        upper_percentile=upper_percentile, column_to_fit="peak_integral",max_reduced_chi2= max_reduced_chi2, log=log)

    #plot the correlations between baseline, sigma and area for each channel, and save the plots to a folder
    plot_waveform_correlations(parquet_file, output_folder=output_folder, channel=None, plot_type='hexbin')





def process_all_parquet_files_in_folder_to_canvas(parquet_folder, output_folder, max_reduced_chi2=2):
    parquet_folder_path = Path(parquet_folder)
    parquet_files = list(parquet_folder_path.rglob("*.parquet"))
    
    if not parquet_files:
        print(f"No Parquet files found.")
        return

    for counter, parquet_file in enumerate(parquet_files, start=1):
        print(f"\nProcessing Parquet file: {counter} / {len(parquet_files)}")
        
        # Create a new process for this specific file
        p = multiprocessing.Process(
            target=process_single_parquet_to_canvas, 
            args=(parquet_file,), 
            kwargs={'output_folder': output_folder, 'max_reduced_chi2': max_reduced_chi2}
        )
        
        p.start() # Start the process
        p.join()  # Pause the main script until this process finishes

        # If the process was killed by the OS (exit code != 0), you'll know exactly which file caused it
        if p.exitcode != 0:
            print(f"WARNING: Process crashed on file {parquet_file} with exit code {p.exitcode}")



def has_events_tree(filepath):
    """
    Scans a ROOT file and returns True if it contains an 'Events' tree, 
    False otherwise.
    """
    filepath_str = str(filepath)
    try:
        with uproot.open(filepath_str) as f:
            base_keys = [k.split(';')[0] for k in f.keys()]
            return "Events" in base_keys
    except Exception as e:
        print(f"Error reading {filepath_str} for tree check: {e}")
        return False


def create_all_parquet_files_from_root_folder(root_folder, parquet_folder):
    """
    Recursively searches for all ROOT files, checks their internal tree structure, 
    processes each file with the appropriate function, and names the Parquet 
    files based on their type.
    """
    root_folder_path = Path(root_folder)
    parquet_folder_path = Path(parquet_folder)
    
    # Find all ROOT files recursively
    root_files = list(root_folder_path.rglob("*.root"))
    
    if not root_files:
        print(f"No ROOT files found in {root_folder}.")
        return

    print(f"Found {len(root_files)} ROOT files. Processing...")
    no_of_files = 0
    
    for counter, root_file in enumerate(root_files, start=1):
        # 1. DETERMINE FILE TYPE FIRST
        # We must know the type to know what the output file will be named
        is_type_a = has_events_tree(root_file)
        
        if is_type_a:
            # Type A: Named after the parent folder
            original_name = root_file.parent.name
            file_type_msg = "Type A ('Events' tree)"
        else:
            # Type B: Named after the file itself
            original_name = root_file.stem
            file_type_msg = "Type B ('Data_R' tree)"
            
        expected_parquet_path = parquet_folder_path / f"{original_name}.parquet"
        
        # 2. SKIP ALREADY PROCESSED FILES
        if expected_parquet_path.exists():
            print(f"\n[{counter}/{len(root_files)}] Skipping {original_name} - Parquet file already exists.")
            no_of_files += 1 
            continue
            
        print(f"\n[{counter}/{len(root_files)}] Processing {file_type_msg}: {root_file.name} -> {original_name}.parquet")
        
        # 3. ROUTE TO CORRECT FUNCTION & ERROR HANDLING
        try:
            if is_type_a:
                analyse_root_data(root_file, parquet_folder=parquet_folder) 
            else:
                analyse_root_data_type_b(root_file, parquet_folder=parquet_folder)
                
            no_of_files += 1 
            
        except Exception as e:
            print(f"FAILED to process {root_file.name}. Error: {e}")
            continue 
            
    print(f"\n{no_of_files} parquet files processed or created.")
            







def fit_single_channel_data(raw_data, channel, bins=500, num_peaks_to_fit=18, max_reduced_chi2=None, lower_percentile=2.0, upper_percentile=99.9, actual_peaks_to_fit=4): 
    raw_data = np.array(raw_data.dropna(), dtype=float) 
    raw_data = -1 * raw_data  # Invert the data for fitting
    # 1. Cut the long tails to focus bins on the actual accumulation
    
    
    min_val = np.percentile(raw_data, lower_percentile)
    max_val = np.percentile(raw_data, upper_percentile)
    raw_data = raw_data[(raw_data >= min_val) & (raw_data <= max_val)]
    

    print(f"Processing {len(raw_data)} data points for {channel} after tail cut...") 

    counts, bin_edges = np.histogram(raw_data, bins=bins) 
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 
    dx = bin_centers[1] - bin_centers[0] 
    
    y_err = np.maximum(np.sqrt(counts), 1.0) + (0.02 * counts)
    
    fit_successful = False
    popt, pcov = None, None
    peak_params = [] 
    expected_counts = None  
    
    reduced_chi_square = np.nan
    chi_square = np.nan
    ndf = 0

    smoothed_counts = gaussian_filter1d(counts, sigma=2) 
    peaks, _ = find_peaks( 
        smoothed_counts,  
        prominence=np.max(smoothed_counts) * 0.015,  # Increased to 1.5% of max peak
        distance=5 # Prevent detecting adjacent bins as separate peaks
    )
    
    global_max_bin = np.argmax(counts) 
    if not any(abs(p - global_max_bin) <= 5 for p in peaks): 
        peaks = np.append(peaks, global_max_bin) 
        peaks = np.sort(peaks) 

    if len(peaks) == 0: 
        print("Failed to find any peaks. Skipping fit.")
    else:
        idx_0 = peaks[0] 
        mu_0_guess = bin_centers[idx_0] 
        
        if len(peaks) > 1: 
            # DYNAMIC GAIN: Calculate all spacings and take the median
            peak_locations = bin_centers[peaks]
            peak_spacings = np.diff(peak_locations)
            gain_guess = np.median(peak_spacings)
        else: 
            gain_guess = (bin_centers[-1] - bin_centers[0]) / (num_peaks_to_fit / 2)

        sigma_0_guess = gain_guess * 0.1  
        sigma_1_guess = gain_guess * 0.05

        p0, lower_bounds, upper_bounds = [], [], []

        A_bg_guess = np.max(counts) * 0.1 
        lambda_bg_guess = 1.0 / max(np.abs(np.mean(raw_data)), 1.0) 
        
        # Base parameters (Indices 0-5)
        p0.extend([A_bg_guess, lambda_bg_guess, mu_0_guess, sigma_0_guess, gain_guess, sigma_1_guess]) 
        # Constrain the pedestal (mu_0) to stay within +/- 1/4 of a gain step from its guess
        lower_bounds.extend([0, 0, mu_0_guess - (gain_guess * 0.25), 1e-12, gain_guess * 0.8, 1e-12]) 
        upper_bounds.extend([np.max(counts) * 0.5, 5000, mu_0_guess + (gain_guess * 0.25), gain_guess * 1.2, gain_guess * 1.2, gain_guess * 1.2])

        # Add Baseline Constant Bounds (Index 6)
        p0.append(1.0)
        lower_bounds.append(0.0)
        upper_bounds.append(np.max(counts) * 0.1)

        # Append amplitude guesses for all requested peaks
        for n in range(num_peaks_to_fit): 
            expected_mu = mu_0_guess + (n * gain_guess) 
            if bin_centers[0] <= expected_mu <= bin_centers[-1]: 
                closest_bin_idx = np.abs(bin_centers - expected_mu).argmin() 
                A_guess = counts[closest_bin_idx] 
            else: 
                A_guess = 1e-9  
                
            p0.append(A_guess) 
            lower_bounds.append(0.0) 
            # FIX: Prevent the amplitude from exploding higher than the tallest peak in your data
            upper_bounds.append(np.max(counts) * 1.5)

        p0 = np.clip(np.array(p0, dtype=float), np.array(lower_bounds, dtype=float) + 1e-10, np.array(upper_bounds, dtype=float) - 1e-10) 

        # 2. Mask the data to only fit the first 'actual_peaks_to_fit' peaks
        max_fit_x = mu_0_guess + ((actual_peaks_to_fit - 1) * gain_guess) + (0.6 * gain_guess)
        fit_mask = bin_centers <= max_fit_x
        
        bin_centers_fit = bin_centers[fit_mask]
        counts_fit = counts[fit_mask]
        y_err_fit = y_err[fit_mask]
        
        # Limit p0 and bounds to the base parameters + amplitudes for the peaks we are actually fitting
        params_to_use = 7 + actual_peaks_to_fit
        p0_fit = p0[:params_to_use]
        lower_bounds_fit = lower_bounds[:params_to_use]
        upper_bounds_fit = upper_bounds[:params_to_use]

        try: 
            popt_temp, pcov_temp = curve_fit( 
                fit_model_physics,  
                bin_centers_fit,  
                counts_fit,  
                p0=p0_fit, 
                sigma=y_err_fit,          
                absolute_sigma=True,  
                bounds=(lower_bounds_fit, upper_bounds_fit), 
                maxfev=25000  
            ) 
            
            A_bg, lambda_bg, mu_0, sigma_0, gain, sigma_1, C = popt_temp[0:7]
            
            # --- NEW: Reconstruct ALL peaks for plotting ---
            full_popt = list(popt_temp)
            
            # We fitted 'actual_peaks_to_fit'. Now estimate the remaining amplitudes 
            # directly from the histogram data using the newly fitted gain!
            for n in range(actual_peaks_to_fit, num_peaks_to_fit):
                expected_mu = mu_0 + (n * gain)
                if bin_centers[0] <= expected_mu <= bin_centers[-1]:
                    closest_bin_idx = np.abs(bin_centers - expected_mu).argmin()
                    A_extrapolated = counts[closest_bin_idx]
                else:
                    A_extrapolated = 1e-9
                full_popt.append(A_extrapolated)
                
            # Evaluate expected counts across the ENTIRE histogram using ALL peaks
            expected_counts = fit_model_physics(bin_centers, *full_popt)
            # -----------------------------------------------
            
            # Calculate Chi-Square specifically on the fitted region
            expected_counts_fit = fit_model_physics(bin_centers_fit, *popt_temp)
            variances_fit = y_err_fit ** 2
            
            chi_square = np.sum(((counts_fit - expected_counts_fit) ** 2) / variances_fit)
            ndf = len(counts_fit) - len(popt_temp)
            reduced_chi_square = chi_square / ndf if ndf > 0 else np.nan
            
            if max_reduced_chi2 is not None and (reduced_chi_square > max_reduced_chi2):
                print(f"Fit rejected: Reduced χ² ({reduced_chi_square:.2f}) > threshold ({max_reduced_chi2}).")
            if max_reduced_chi2 is not None and (0.8 > reduced_chi_square):
                print ((f"Fit rejected: Reduced χ² (0.8 > {reduced_chi_square:.2f}) > threshold."))

            else:
                print("Physics Curve fitting converged successfully!") 
                fit_successful = True
                popt = full_popt  # Return the full parameter set so your plotting code works
                pcov = pcov_temp
                
                # Extrapolate peak parameters for all requested peaks
                for n in range(num_peaks_to_fit):
                    mu_n = mu_0 + n * gain
                    sigma_n = np.sqrt(sigma_0**2 + n * sigma_1**2)
                    peak_params.append([mu_n, sigma_n])
            
        except Exception as e: 
            print(f"Optimal parameters not found: {e}")

    return {
        "bin_centers": bin_centers, "bin_edges": bin_edges, "counts": counts,
        "y_err": y_err, "expected_counts": expected_counts, "popt": popt,
        "pcov": pcov, "peak_params": np.array(peak_params) if peak_params else np.array([]),
        "fit_successful": fit_successful, "chi_square": chi_square,
        "reduced_chi_square": reduced_chi_square, "ndf": ndf
    }


'''
def fit_single_channel_data(raw_data, channel, bins=500, num_peaks_to_fit=18, max_reduced_chi2=None, lower_percentile=2.0, upper_percentile=99.9, actual_peaks_to_fit=4): 
    raw_data = np.array(raw_data.dropna(), dtype=float) 
    raw_data = -1 * raw_data  # Invert the data for fitting
    # 1. Cut the long tails to focus bins on the actual accumulation
    min_val = np.percentile(raw_data, lower_percentile)
    max_val = np.percentile(raw_data, upper_percentile)
    raw_data = raw_data[(raw_data >= min_val) & (raw_data <= max_val)]

    print(f"Processing {len(raw_data)} data points for {channel} after tail cut...") 

    counts, bin_edges = np.histogram(raw_data, bins=bins) 
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 
    dx = bin_centers[1] - bin_centers[0] 
    
    y_err = np.maximum(np.sqrt(counts), 1.0) + (0.02 * counts)
    
    fit_successful = False
    popt, pcov = None, None
    peak_params = [] 
    expected_counts = None  
    
    reduced_chi_square = np.nan
    chi_square = np.nan
    ndf = 0

    smoothed_counts = gaussian_filter1d(counts, sigma=1) 
    peaks, _ = find_peaks( 
        smoothed_counts,  
        prominence=np.max(smoothed_counts) * 0.01,  
        distance=4 
    ) 
    
    global_max_bin = np.argmax(counts) 
    if not any(abs(p - global_max_bin) <= 5 for p in peaks): 
        peaks = np.append(peaks, global_max_bin) 
        peaks = np.sort(peaks) 

    if len(peaks) == 0: 
        print("Failed to find any peaks. Skipping fit.")
    else:
        idx_0 = peaks[0] 
        mu_0_guess = bin_centers[idx_0] 
        
        if len(peaks) > 1: 
            gain_guess = bin_centers[peaks[1]] - bin_centers[peaks[0]] 
        else: 
            gain_guess = (bin_centers[-1] - bin_centers[0]) / (num_peaks_to_fit / 2) 

        sigma_0_guess = max(dx, 1e-9) 
        sigma_1_guess = sigma_0_guess * 0.5  

        p0, lower_bounds, upper_bounds = [], [], []

        A_bg_guess = np.max(counts) * 0.1 
        lambda_bg_guess = 1.0 / max(np.abs(np.mean(raw_data)), 1.0) 
        
        # Base parameters (Indices 0-5)
        p0.extend([A_bg_guess, lambda_bg_guess, mu_0_guess, sigma_0_guess, gain_guess, sigma_1_guess]) 
        lower_bounds.extend([0, 0, bin_centers[0] - dx*10, 1e-12, dx, 1e-12]) 
        upper_bounds.extend([np.max(counts) * 0.5, 5000, bin_centers[-1], gain_guess * 1.2, gain_guess * 5, gain_guess * 1.2]) 

        # Add Baseline Constant Bounds (Index 6)
        p0.append(1.0)
        lower_bounds.append(0.0)
        upper_bounds.append(np.max(counts) * 0.1)

        # Append amplitude guesses for all requested peaks
        for n in range(num_peaks_to_fit): 
            expected_mu = mu_0_guess + (n * gain_guess) 
            if bin_centers[0] <= expected_mu <= bin_centers[-1]: 
                closest_bin_idx = np.abs(bin_centers - expected_mu).argmin() 
                A_guess = counts[closest_bin_idx] 
            else: 
                A_guess = 1e-9  
                
            p0.append(A_guess) 
            lower_bounds.append(0.0) 
            upper_bounds.append(np.inf) 

        p0 = np.clip(np.array(p0, dtype=float), np.array(lower_bounds, dtype=float) + 1e-10, np.array(upper_bounds, dtype=float) - 1e-10) 

        # 2. Mask the data to only fit the first 'actual_peaks_to_fit' peaks
        max_fit_x = mu_0_guess + ((actual_peaks_to_fit - 1) * gain_guess) + (1.5 * gain_guess)
        fit_mask = bin_centers <= max_fit_x
        
        bin_centers_fit = bin_centers[fit_mask]
        counts_fit = counts[fit_mask]
        y_err_fit = y_err[fit_mask]
        
        # Limit p0 and bounds to the base parameters + amplitudes for the peaks we are actually fitting
        params_to_use = 7 + actual_peaks_to_fit
        p0_fit = p0[:params_to_use]
        lower_bounds_fit = lower_bounds[:params_to_use]
        upper_bounds_fit = upper_bounds[:params_to_use]

        try: 
            popt_temp, pcov_temp = curve_fit( 
                fit_model_physics,  
                bin_centers_fit,  
                counts_fit,  
                p0=p0_fit, 
                sigma=y_err_fit,          
                absolute_sigma=True,  
                bounds=(lower_bounds_fit, upper_bounds_fit), 
                maxfev=25000  
            ) 
            
            A_bg, lambda_bg, mu_0, sigma_0, gain, sigma_1, C = popt_temp[0:7]
            
            # --- NEW: Reconstruct ALL peaks for plotting ---
            full_popt = list(popt_temp)
            
            # We fitted 'actual_peaks_to_fit'. Now estimate the remaining amplitudes 
            # directly from the histogram data using the newly fitted gain!
            for n in range(actual_peaks_to_fit, num_peaks_to_fit):
                expected_mu = mu_0 + (n * gain)
                if bin_centers[0] <= expected_mu <= bin_centers[-1]:
                    closest_bin_idx = np.abs(bin_centers - expected_mu).argmin()
                    A_extrapolated = counts[closest_bin_idx]
                else:
                    A_extrapolated = 1e-9
                full_popt.append(A_extrapolated)
                
            # Evaluate expected counts across the ENTIRE histogram using ALL peaks
            expected_counts = fit_model_physics(bin_centers, *full_popt)
            # -----------------------------------------------
            
            # Calculate Chi-Square specifically on the fitted region
            expected_counts_fit = fit_model_physics(bin_centers_fit, *popt_temp)
            variances_fit = y_err_fit ** 2
            
            chi_square = np.sum(((counts_fit - expected_counts_fit) ** 2) / variances_fit)
            ndf = len(counts_fit) - len(popt_temp)
            reduced_chi_square = chi_square / ndf if ndf > 0 else np.nan
            
            if max_reduced_chi2 is not None and reduced_chi_square > max_reduced_chi2:
                print(f"Fit rejected: Reduced χ² ({reduced_chi_square:.2f}) > threshold ({max_reduced_chi2}).")
            else:
                print("Physics Curve fitting converged successfully!") 
                fit_successful = True
                popt = full_popt  # Return the full parameter set so your plotting code works
                pcov = pcov_temp
                
                # Extrapolate peak parameters for all requested peaks
                for n in range(num_peaks_to_fit):
                    mu_n = mu_0 + n * gain
                    sigma_n = np.sqrt(sigma_0**2 + n * sigma_1**2)
                    peak_params.append([mu_n, sigma_n])
            
        except Exception as e: 
            print(f"Optimal parameters not found: {e}")

    return {
        "bin_centers": bin_centers, "bin_edges": bin_edges, "counts": counts,
        "y_err": y_err, "expected_counts": expected_counts, "popt": popt,
        "pcov": pcov, "peak_params": np.array(peak_params) if peak_params else np.array([]),
        "fit_successful": fit_successful, "chi_square": chi_square,
        "reduced_chi_square": reduced_chi_square, "ndf": ndf
    }



def compare_parquet_data(parquet_files, target_channel, bins=500, num_peaks=18, max_reduced_chi2=None, output_folder=None, log=False):
    """
    Reads multiple parquet files, isolates a channel, calculates the fit.
    Plots each file individually to disk, then streams them onto a gigantic horizontal canvas.
    """
    total_files = len(parquet_files)
    cols_to_keep = ['channel', 'peak_integral', 'baseline', 'std_dev']
    
    # Prepare save directories
    if output_folder is not None:
        save_dir = Path(output_folder)
    else:
        save_dir = Path(comparisons_folder)
        
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a temporary folder for the individual PNG slices
    temp_dir = save_dir / "temp_slices"
    temp_dir.mkdir(exist_ok=True)
    
    final_save_path = save_dir / f"{target_channel}_gigantic_canvas.pdf"
    
    print(f"Starting plot generation. Total files: {total_files}.")
    
    temp_image_paths = []
    colors = plt.cm.tab10(np.linspace(0, 1, total_files))

    # ==========================================
    # PHASE 1: Create Individual Plot Columns
    # ==========================================
    for col_idx, filepath in enumerate(parquet_files):
        print(f"--- Processing {col_idx + 1} / {total_files}: {Path(filepath).name} ---")
        
        try:
            # MEMORY FIX 2: Disable pyarrow memory mapping
            df_chan = pd.read_parquet(
                filepath, 
                columns=cols_to_keep,
                filters=[('channel', '==', target_channel)],
                engine='pyarrow',
                memory_map=False
            )
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            continue
            
        if df_chan.empty:
            print(f"  -> Warning: Channel '{target_channel}' not found in this file. Skipping.")
            continue
            
        color = colors[col_idx]
        
        full_areas = df_chan['peak_integral']
        fit_results = fit_single_channel_data(full_areas, target_channel, bins=bins, num_peaks_to_fit=num_peaks, max_reduced_chi2=max_reduced_chi2)
        
        MAX_POINTS = 1_000_000
        if len(df_chan) > MAX_POINTS:
            plot_df = df_chan.sample(n=MAX_POINTS, random_state=42)
        else:
            plot_df = df_chan
            
        areas = plot_df['peak_integral'].values
        baselines = plot_df['baseline'].values
        sigmas = plot_df['std_dev'].values
        
        fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(7, 18), dpi=300)
        
        metadata = parse_folder_metadata(filepath)

        if metadata.get("is_dark_count", False):
            conditions = [metadata['temp_c'], 0, metadata['bias_voltage'], metadata["amp_min_v"], metadata["amp_max_v"]]
        else:
            conditions = [metadata['temp_c'], metadata['led_voltage'], metadata['bias_voltage'], metadata["amp_min_v"], metadata["amp_max_v"]]
        
        hex_gridsize = 80
        hex_cmap = 'viridis'

        # --- Row 0: Histogram ---
        ax0 = axes[0]
        ax0.fill_between(fit_results['bin_centers'], fit_results['counts'], y2=0.1, step='mid', color=color, alpha=0.4)
        ax0.plot(fit_results['bin_centers'], fit_results['counts'], drawstyle='steps-mid', color=color, alpha=0.9, linewidth=1.2)
        
        if fit_results['fit_successful']:
            ax0.plot(fit_results['bin_centers'], fit_results['expected_counts'], color='red', linestyle='--', 
                     label=f"Fit (\u03C7\u00B2/ndf = {fit_results['reduced_chi_square']:.2f})")

        if log:
            ax0.set_yscale('log')
            ax0.set_ylim(bottom=0.1)
        else:
            ax0.set_ylim(bottom=0)


        if metadata.get("is_dark_count", False):
            ax0.set_title(f"Histogram & Fit - {target_channel}\n"
                f"{float(conditions[0]):.1f} °C / {float(conditions[0])+273.15:.1f} K\n"
                f"No LED, Dark Count measurement | SiPM: {float(conditions[2]):.1f} V")


        else:
            ax0.set_title(f"Histogram & Fit - {target_channel}\n"
                f"{float(conditions[0]):.1f} °C / {float(conditions[0])+273.15:.1f} K\n"
                f"LED: {float(conditions[1]):.2f} V | SiPM: {float(conditions[2]):.1f} V")


        ax0.set_xlabel("Area / Integral [a.u.]")
        ax0.set_ylabel("Counts")
        ax0.legend(loc="best")
        ax0.grid(True, linestyle=':', alpha=0.6)

        # --- Rows 1-3: Hexbins ---
        ax1 = axes[1]
        hb1 = ax1.hexbin(areas, baselines, gridsize=hex_gridsize, cmap=hex_cmap, mincnt=1, rasterized=True)
        fig.colorbar(hb1, ax=ax1, label='Counts')
        ax1.set_title("Baseline vs Area", fontweight='bold')
        ax1.set_xlabel("Area")
        ax1.set_ylabel("Baseline")

        ax2 = axes[2]
        hb2 = ax2.hexbin(areas, sigmas, gridsize=hex_gridsize, cmap=hex_cmap, mincnt=1, rasterized=True)
        fig.colorbar(hb2, ax=ax2, label='Counts')
        ax2.set_title("Sigma vs Area", fontweight='bold')
        ax2.set_xlabel("Area")
        ax2.set_ylabel("Sigma")

        ax3 = axes[3]
        hb3 = ax3.hexbin(sigmas, baselines, gridsize=hex_gridsize, cmap=hex_cmap, mincnt=1, rasterized=True)
        fig.colorbar(hb3, ax=ax3, label='Counts')
        ax3.set_title("Baseline vs Sigma", fontweight='bold')
        ax3.set_xlabel("Sigma")
        ax3.set_ylabel("Baseline")

        plt.tight_layout()
        
        # Save this single column as a temporary PNG
        temp_img_path = temp_dir / f"col_{col_idx:03d}.png"
        fig.savefig(temp_img_path, format='png', bbox_inches='tight')
        temp_image_paths.append(temp_img_path)

        # MEMORY FIX 3: Aggressive Teardown
        fig.clf()
        plt.close(fig)
        plt.close('all')
        
        del df_chan, plot_df, full_areas, areas, baselines, sigmas
        gc.collect()
        # Updated to use 'py' instead of 'pa'
        py.default_memory_pool().release_unused()

    # ==========================================
    # PHASE 2: Stitch Images into Gigantic Canvas
    # ==========================================
    if not temp_image_paths:
        print(f"\nERROR: Target channel '{target_channel}' was not found in ANY of the provided files.")
        try:
            temp_dir.rmdir()
        except OSError:
            pass
        return 
        
    print("\n>>> All files processed. Stitching individual plots into a gigantic canvas... <<<")
    
    # Disable PIL protection for massive images (prevents DecompressionBombError)
    Image.MAX_IMAGE_PIXELS = None
    
    # 1. Calculate dimensions WITHOUT keeping files open in memory
    total_width = 0
    max_height = 0
    for p in temp_image_paths:
        with Image.open(p) as img:
            total_width += img.size[0]
            max_height = max(max_height, img.size[1])
            
    # 2. Allocate the single giant canvas once
    gigantic_canvas = Image.new('RGB', (total_width, max_height), color='white')
    
    # 3. Open, paste, and close one image at a time
    x_offset = 0
    for p in temp_image_paths:
        with Image.open(p) as img:
            gigantic_canvas.paste(img, (x_offset, 0))
            x_offset += img.size[0]
            
    # Save the final stitched image directly as PDF
    gigantic_canvas.save(final_save_path, format="PDF", resolution=300.0)
    print(f"\nSUCCESS! Gigantic canvas PDF saved to: {final_save_path}")
    
    # Cleanup: Delete the temporary images and the temp folder
    for p in temp_image_paths:
        p.unlink(missing_ok=True)
    try:
        temp_dir.rmdir()
    except OSError:
        pass
'''



# ==========================================
# HELPER FUNCTIONS
# ==========================================


def get_parquet_files(folder_path):
    folder = Path(folder_path)
    
    if not folder.is_dir():
        print(f"Error: The folder '{folder_path}' does not exist.")
        return []

    # List comprehension that checks if the item is a file and has the .parquet suffix
    parquet_filepaths = [
        str(item.absolute()) for item in folder.iterdir() 
        if item.is_file() and item.suffix.lower() == '.parquet'
    ]
    
    return parquet_filepaths




def get_matching_root_files(parquet_files, root_base_folder):
    """
    Finds corresponding ROOT files for Parquet files. 
    Matches against both File Type A (named after folder) and Type B (named after file).
    Allows Parquet files to have extra suffixes separated by underscores.
    """
    root_base = Path(root_base_folder)
    
    # 1. Build TWO indices of the .root files
    print(f"Scanning for .root files in {root_base}...")
    root_index_by_folder = defaultdict(list)
    root_index_by_file = defaultdict(list)
    
    for root_file in root_base.rglob("*.root"):
        root_index_by_folder[root_file.parent.name].append(str(root_file))
        root_index_by_file[root_file.stem].append(str(root_file))

    # 2. Match Parquet files to the found ROOT files
    matched_root_files = []
    
    for pq_file in parquet_files:
        pq_stem = Path(pq_file).stem  # e.g., 'data_run_123_v2'
        
        # Helper function to check both dictionaries
        def find_match(candidate_name):
            # Check Type B first (exact file name match is a stronger signal)
            if candidate_name in root_index_by_file:
                return root_index_by_file[candidate_name]
            # Then check Type A (folder name match)
            if candidate_name in root_index_by_folder:
                return root_index_by_folder[candidate_name]
            return None
            
        # Check exact match first
        available_roots = find_match(pq_stem)
        
        # If no exact match, progressively drop suffixes from the right
        # e.g., 'data_run_123_v2' -> tries 'data_run_123'
        if not available_roots:
            parts = pq_stem.split('_')
            for i in range(len(parts) - 1, 0, -1):
                candidate = '_'.join(parts[:i])
                available_roots = find_match(candidate)
                if available_roots:
                    break  # Found a match after stripping suffixes!

        # 3. Retrieve the matched files
        if available_roots:
            if len(available_roots) > 1:
                print(f"Warning: Multiple .root files found for '{pq_stem}'. Using {available_roots[0]}")
                
            matched_root_files.append(available_roots[0])
        else:
            print(f"Warning: Could not find a matching .root file or folder for '{pq_stem}'.")
            matched_root_files.append(None)
            
    return matched_root_files


# ==========================================
# MAIN PLOTTING FUNCTION
# ==========================================




def plot_peak_area_histogram(ax, fit_results, color, metadata, target_channel, log_scale=False):
    """Plots the primary peak area histogram and fit line."""
    centers, counts = fit_results['bin_centers'], fit_results['counts']
    
    ax.fill_between(centers, counts, y2=0.1, step='mid', color=color, alpha=0.4)
    ax.plot(centers, counts, drawstyle='steps-mid', color=color, alpha=0.9, linewidth=1.2)
    
    if fit_results.get('fit_successful'):
        label = f"Fit (\u03C7\u00B2/ndf = {fit_results['reduced_chi_square']:.2f})"
        ax.plot(centers, fit_results['expected_counts'], color='red', linestyle='--', label=label)

    ax.set_yscale('log') if log_scale else ax.set_ylim(bottom=0)
    if log_scale: 
        ax.set_ylim(bottom=0.1)

    # Extract metadata cleanly
    temp_c = float(metadata.get('temp_c', 0))
    bias_v = float(metadata.get('bias_voltage', 0))
    led_v = float(metadata.get('led_voltage', 0))
    mode_prefix = f"{metadata.get('mode', '')} ".strip()

    title_lines = [
        f"Histogram & Fit - {mode_prefix} {target_channel}",
        f"{temp_c:.1f} °C / {temp_c + 273.15:.1f} K"
    ]
    
    if metadata.get("is_dark_count", False):
        title_lines.append(f"No LED, Dark Count | SiPM: {bias_v:.1f} V")
    else:
        title_lines.append(f"LED: {led_v:.2f} V | SiPM: {bias_v:.1f} V")

    ax.set_title("\n".join(title_lines))
    ax.set_xlabel("Area / Integral [a.u.]")
    ax.set_ylabel("Counts")
    ax.legend(loc="best")
    ax.grid(True, linestyle=':', alpha=0.6)

# ==========================================
# 1. MATH & FITTING HELPERS
# ==========================================
def gaussian(x, a, mu, sigma):
    """1D Gaussian function for curve fitting."""
    return a * np.exp(-0.5 * ((x - mu) / (sigma + 1e-10))**2)

def fit_and_plot_gaussian_hist(ax, data, title, xlabel, color, log_scale=False):
    """Plots a histogram and fits a Gaussian to it on the provided matplotlib axis."""
    counts, bins, _ = ax.hist(data, bins=500, color=color, alpha=0.7, label='Data')
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    try:
        p0 = [np.max(counts), np.mean(data), np.std(data)]
        popt, _ = curve_fit(gaussian, bin_centers, counts, p0=p0)
        x_fit = np.linspace(bins[0], bins[-1], 500)
        ax.plot(x_fit, gaussian(x_fit, *popt), color='red', linestyle='--', label=rf'Fit ($\mu={popt[1]:.2f}$)')
    except Exception as e:
        print(f"Warning: Gaussian fit failed for {title} - {e}")
        
    if log_scale:
        ax.set_yscale('log')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Counts")
    ax.legend(loc="best")
    ax.grid(True, linestyle=':', alpha=0.6)



# ==========================================
# 2. PLOTTING HELPERS
# ==========================================
def plot_hexbin(ax, x, y, title, xlabel, ylabel, fig):
    """Plots a standardized hexbin on the provided axis."""
    hb = ax.hexbin(x, y, gridsize=80, cmap='viridis', mincnt=1, rasterized=True)
    fig.colorbar(hb, ax=ax, label='Counts')
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)



def plot_signals(axes, root_filepath, target_channel, num_samples, device, valid_indices=None):
    """Extracts ROOT signals and plots them across the provided axes. 
    Automatically adapts to Type A (Events, negative peaks) and Type B (Data_R, positive peaks)."""
    
    if not (root_filepath and Path(root_filepath).exists()):
        for ax in axes: ax.text(0.5, 0.5, "File missing", ha='center')
        return

    filepath_str = str(root_filepath)
    is_positive_peak = False

    # 1. FILE TYPE INSPECTION AND DATA EXTRACTION
    try:
        with uproot.open(filepath_str) as f:
            base_keys = [k.split(';')[0] for k in f.keys()]
            
            # --- TYPE A: Events Tree ---
            if 'Events' in base_keys:
                tree = f['Events']
                if target_channel in tree.keys():
                    samples_matrix = tree[target_channel].array(library="np")
                else:
                    for ax in axes: ax.text(0.5, 0.5, "Channel not found", ha='center')
                    return
                    
            # --- TYPE B: Data_R Trees ---
            elif any(k.startswith('Data_R') for k in base_keys):
                is_positive_peak = True
                data_r_keys = [k for k in f.keys() if k.startswith("Data_R")]
                
                # Convert "ch0" to 0 for filtering
                ch_num = int(target_channel.replace("ch", ""))
                
                samples_list = []
                for k in data_r_keys:
                    for chunk in uproot.iterate({filepath_str: k}, filter_name=["Samples", "Channel"], library="np", step_size="100 MB"):
                        mask = (chunk["Channel"] == ch_num)
                        waves = chunk["Samples"][mask]
                        if len(waves) > 0:
                            if waves.dtype == 'O': waves = np.vstack(waves)
                            samples_list.append(waves)
                
                if not samples_list:
                    for ax in axes: ax.text(0.5, 0.5, "Channel not found", ha='center')
                    return
                samples_matrix = np.vstack(samples_list)
                
            else:
                for ax in axes: ax.text(0.5, 0.5, "Unrecognized ROOT structure", ha='center')
                return
                
            # Randomization and Indexing
            total_rows = len(samples_matrix)
            if valid_indices is not None and len(valid_indices) > 0:
                # Filter out any indices that might exceed our extracted array
                valid_indices = [idx for idx in valid_indices if idx < total_rows]
                search_indices = np.array(valid_indices) if valid_indices else np.arange(total_rows)
            else:
                search_indices = np.arange(total_rows)
                
            np.random.shuffle(search_indices)
            
    except Exception as e:
        print(f"Error reading ROOT file {root_filepath}: {e}")
        return

    # 2. PLOTTING LOOP
    plotted_count = 0
    
    for row_idx in search_indices:
        if plotted_count >= len(axes) or plotted_count >= num_samples:
            break
            
        signal = np.asarray(samples_matrix[row_idx], dtype=float)
        valid_mask = ~np.isnan(signal)
        sig = signal[valid_mask]
        
        if len(sig) < 50:
            continue

        t = np.arange(len(sig))

        if device is not None:
            bkg_start, bkg_end = baseline_start_4, baseline_end_4
            start_indx, end_indx = max(0, peak_start_4), min(len(t), peak_end_4)
        else:
            bkg_start, bkg_end = baseline_start_64, baseline_end_64
            start_indx, end_indx = max(0, peak_start_64), min(len(t), peak_end_64)

        # Skip if array is too short for the windows
        if bkg_end > len(sig) or end_indx > len(sig):
            continue

        bkg_sig = sig[bkg_start:bkg_end]
        baseline_val = np.mean(bkg_sig) if len(bkg_sig) > 0 else 0
        sigma_val = np.std(bkg_sig, ddof=1) if len(bkg_sig) > 1 else 0.0

        sig_window_initial = sig[start_indx:end_indx]
        if len(sig_window_initial) == 0:
            continue
            
        # 3. DYNAMIC PEAK FINDING (Positive vs Negative)
        if is_positive_peak:
            # --- POSITIVE PEAK LOGIC (Type B) ---
            peak_val = np.max(sig_window_initial)
            if peak_val < (baseline_val + 3 * sigma_val): 
                continue 
                
            peak_idx = start_indx + np.argmax(sig_window_initial)
            
            ext_start, ext_end = peak_idx, peak_idx
            while ext_start > 0 and sig[ext_start] > baseline_val: ext_start -= 1
            while ext_end < len(sig) - 1 and sig[ext_end] > baseline_val: ext_end += 1
            
            fill_condition = (sig[ext_start:ext_end + 1] > baseline_val)
        else:
            # --- NEGATIVE PEAK LOGIC (Type A) ---
            peak_val = np.min(sig_window_initial)
            if peak_val > (baseline_val - 3 * sigma_val): 
                continue 
                
            peak_idx = start_indx + np.argmin(sig_window_initial)
            
            ext_start, ext_end = peak_idx, peak_idx
            while ext_start > 0 and sig[ext_start] < baseline_val: ext_start -= 1
            while ext_end < len(sig) - 1 and sig[ext_end] < baseline_val: ext_end += 1
            
            fill_condition = (sig[ext_start:ext_end + 1] < baseline_val)

        # 4. RENDER PLOT
        ax_sig = axes[plotted_count]
        
        ax_sig.plot(t, sig, label=f'Signal Row {row_idx}', color='blue')
        ax_sig.axhline(y=baseline_val, color='green', linestyle='--', label='Baseline')
        ax_sig.axhline(y=baseline_val + sigma_val, color='orange', linestyle='--', label='+/- 1 Sigma')
        ax_sig.axhline(y=baseline_val - sigma_val, color='orange', linestyle='--')
        
        ax_sig.axhline(y=baseline_val + 3*sigma_val, color='red', linestyle=':', label='+/- 3 Sigma')
        ax_sig.axhline(y=baseline_val - 3*sigma_val, color='red', linestyle=':')
        
        if 0 <= start_indx < len(t):
            ax_sig.axvline(x=t[start_indx], color='black', linestyle='--', label='Window')
        if 0 <= end_indx - 1 < len(t):
            ax_sig.axvline(x=t[end_indx - 1], color='black', linestyle='--')

        t_window = t[ext_start:ext_end + 1]
        sig_window = sig[ext_start:ext_end + 1]

        if len(t_window) > 0:
            ax_sig.fill_between(t_window, sig_window, baseline_val, where=fill_condition, 
                              color='purple', alpha=0.3, interpolate=True)
    
        ax_sig.set_title(f"Sample {plotted_count+1} (Row {row_idx})", fontsize=10)
        ax_sig.set_xlabel('Samples')
        ax_sig.set_ylabel('ADC')
        ax_sig.legend(loc='upper right', fontsize='small')
        ax_sig.grid(True)
        
        plotted_count += 1

    # Fill any remaining axes with blank text
    for i in range(plotted_count, len(axes)):
        axes[i].text(0.5, 0.5, "No more +/-3 sigma samples available", ha='center', va='center', transform=axes[i].transAxes)


# ==========================================
# 3. IMAGE STITCHING
# ==========================================
def stitch_images_to_pdf(image_paths, output_path):
    """Combines a list of image paths horizontally into one giant PDF."""
    if not image_paths: return
    Image.MAX_IMAGE_PIXELS = None
    
    total_width, max_height = 0, 0
    for p in image_paths:
        with Image.open(p) as img:
            total_width += img.size[0]
            max_height = max(max_height, img.size[1])
        
    canvas = Image.new('RGB', (total_width, max_height), color='white')
    
    x_offset = 0
    for p in image_paths:
        with Image.open(p) as img:
            canvas.paste(img, (x_offset, 0))
            x_offset += img.size[0]
        
    canvas.save(output_path, format="PDF", resolution=300.0)

# ==========================================
# 4. MAIN ORCHESTRATOR
# ==========================================

'''
def compare_parquet_and_root_data(
    parquet_files, root_files, target_channel, 
    bins=500, num_peaks=18, max_reduced_chi2=2,
    lower_percentile=2.0, upper_percentile=99.9,
    output_folder=None, log=False, num_samples=5,
    actual_peaks_to_fit=7, device=None
):
    """
    Reads parquet files for histograms/hexbins and corresponding ROOT files for sample signals.
    Plots each file individually to disk (11 rows), then streams them onto a gigantic horizontal canvas.
    """
    if len(parquet_files) != len(root_files):
        raise ValueError("The number of parquet files must match the number of ROOT files.")
        
    total_files = len(parquet_files)
    save_dir = Path(output_folder) if output_folder else Path("comparisons_folder")
    save_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = save_dir / "temp_slices"
    temp_dir.mkdir(exist_ok=True)
    
    final_save_path = save_dir / f"{target_channel}_gigantic_canvas.pdf"
    print(f"Starting plot generation. Total files: {total_files}.")
    
    temp_image_paths = []
    colors = plt.cm.tab10(np.linspace(0, 1, total_files))

    for col_idx, (pq_filepath, root_filepath) in enumerate(zip(parquet_files, root_files)):
        print(f"--- Processing {col_idx + 1} / {total_files}: {Path(pq_filepath).name} ---")
        
        # --- 1. Read Parquet Data ---
        try:
            df_chan = pd.read_parquet(
                pq_filepath, 
                columns=['channel', 'peak_integral', 'baseline', 'std_dev'],
                filters=[('channel', '==', target_channel)],
                engine='pyarrow',
                memory_map=False
            )
        except Exception as e:
            print(f"Error reading {pq_filepath}: {e}")
            continue
            
        if df_chan.empty:
            print(f"  -> Warning: Channel '{target_channel}' not found. Skipping.")
            continue
            
        color = colors[col_idx]
        
        # Get metadata (Assumes parse_folder_metadata is available in your environment)
        metadata = parse_folder_metadata(pq_filepath)
        if metadata.get("is_dark_count", False):
            conditions = [metadata['temp_c'], 0, metadata['bias_voltage'], metadata.get("amp_min_v", 0), metadata.get("amp_max_v", 0)]
        else:
            conditions = [metadata['temp_c'], metadata['led_voltage'], metadata['bias_voltage'], metadata.get("amp_min_v", 0), metadata.get("amp_max_v", 0)]

        # Get peak fit results (Assumes fit_single_channel_data is available in your environment)
        fit_results = fit_single_channel_data(
            df_chan['peak_integral'], target_channel, bins=bins, 
            num_peaks_to_fit=num_peaks, max_reduced_chi2=max_reduced_chi2,
            lower_percentile=lower_percentile, upper_percentile=upper_percentile,
            actual_peaks_to_fit=actual_peaks_to_fit
        )
        
        MAX_POINTS = 1_000_000
        plot_df = df_chan.sample(n=MAX_POINTS, random_state=42) if len(df_chan) > MAX_POINTS else df_chan
            
        areas = plot_df['peak_integral'].values
        baselines = plot_df['baseline'].values
        sigmas = plot_df['std_dev'].values
        
        # Setup Figure: 3 Hists + 5 Signals + 3 Hexbins = 11 Rows
        fig, axes = plt.subplots(nrows=11, ncols=1, figsize=(7, 44), dpi=300)

        # Row 0: Peak Area Histogram
        plot_peak_area_histogram(axes[0], fit_results, color, metadata, conditions, target_channel, log)



        if fit_results.get("fit_successful") and len(fit_results.get("peak_params", [])) > 0:
            first_mus = fit_results["peak_params"][:5, 0]
            
            # 1. Draw the vertical lines
            for mu in first_mus:
                axes[0].axvline(x=mu, color='black', linestyle=':', alpha=0.7)
            
            # 2. Create a combined label string
            mu_text = "\n".join([f"μ{idx}: {mu:.0f}" for idx, mu in enumerate(first_mus)])
            
            # 3. Place the text box in the top right (x=0.98, y=0.80 prevents overlapping the main legend)
            axes[0].text(0.98, 0.80, mu_text, transform=axes[0].transAxes, 
                         va='top', ha='right', fontsize=9,
                         bbox=dict(facecolor='white', alpha=0.9, edgecolor='lightgray', pad=4))
        
        # Rows 1 & 2: Gaussian fit histograms for Baseline and Sigma
        fit_and_plot_gaussian_hist(axes[1], baselines, "Baseline Histogram", "Baseline", 'green', log)
        fit_and_plot_gaussian_hist(axes[2], sigmas, "Sigma Histogram", "Sigma", 'orange', log)

        # Rows 3-7: Sample Signals
        plot_signals(axes[3:8], root_filepath, target_channel, num_samples, device)

        # Rows 8-10: Hexbins
        plot_hexbin(axes[8], areas, baselines, "Baseline vs Area", "Area", "Baseline", fig)
        plot_hexbin(axes[9], areas, sigmas, "Sigma vs Area", "Area", "Sigma", fig)
        plot_hexbin(axes[10], sigmas, baselines, "Baseline vs Sigma", "Sigma", "Baseline", fig)

        plt.tight_layout()
        
        # Save temp image
        temp_img_path = temp_dir / f"col_{col_idx:03d}.png"
        fig.savefig(temp_img_path, format='png', bbox_inches='tight')
        temp_image_paths.append(temp_img_path)

        # Aggressive Teardown
        fig.clf()
        plt.close(fig)
        plt.close('all')
        
        del df_chan, plot_df, areas, baselines, sigmas
        gc.collect()
        py.default_memory_pool().release_unused()

    # Stitching Phase
    if not temp_image_paths:
        print(f"\nERROR: Target channel '{target_channel}' was not found in ANY files.")
        try: temp_dir.rmdir()
        except OSError: pass
        return 
        
    print("\n>>> All files processed. Stitching individual plots into a gigantic canvas... <<<")
    stitch_images_to_pdf(temp_image_paths, final_save_path)
    print(f"\nSUCCESS! Gigantic canvas PDF saved to: {final_save_path}")
    
    # Cleanup Temp Files
    for p in temp_image_paths:
        p.unlink(missing_ok=True)
    try: temp_dir.rmdir()
    except OSError: pass
'''




def compare_parquet_and_root_data(
    parquet_files, root_files, target_channel, 
    bins=500, num_peaks=18, max_reduced_chi2=2,
    lower_percentile=2.0, upper_percentile=99.9,
    plot_output_folder="comparisons_folder",
    parquet_output_folder=None,
    log=False, num_samples=5,
    actual_peaks_to_fit=7, device=None
):
    """
    Reads parquet/ROOT files, plots 11 rows of data per file to temp images, 
    streams them into a single horizontal PDF canvas, and saves a Parquet of mu values and metadata.
    """
    if len(parquet_files) != len(root_files):
        raise ValueError("The number of parquet and ROOT files must match.")
        
    save_dir = Path(plot_output_folder)
    save_dir.mkdir(parents=True, exist_ok=True)
    final_save_path = save_dir / f"{target_channel}_gigantic_canvas.pdf"
    
    # ==========================================
    # NEW: Skip computation if the PDF exists
    # ==========================================
    if final_save_path.exists():
        print(f"Skipping channel '{target_channel}': {final_save_path} already exists.")
        return
    # ==========================================
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(parquet_files)))
    temp_image_paths = []
    
    # --- NEW: List to store our metadata and mu values for each file ---
    summary_data = []

    print(f"Starting plot generation for '{target_channel}'. Total files: {len(parquet_files)}.")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)

        for col_idx, (pq_filepath, root_filepath) in enumerate(zip(parquet_files, root_files)):
            print(f"--- Processing {col_idx + 1} / {len(parquet_files)}: {Path(pq_filepath).name} ---")
            
            try:
                df_chan = pd.read_parquet(
                    pq_filepath, 
                    columns=['channel', 'peak_integral', 'baseline', 'std_dev'],
                    filters=[('channel', '==', target_channel)]
                )
            except Exception as e:
                print(f"Error reading {pq_filepath}: {e}"); continue
                
            if df_chan.empty:
                print(f"  -> Warning: Channel '{target_channel}' not found. Skipping."); continue
            
            metadata = parse_folder_metadata(pq_filepath)
            
            fit_results = fit_single_channel_data(
                df_chan['peak_integral'], target_channel, bins=bins, 
                num_peaks_to_fit=num_peaks, max_reduced_chi2=max_reduced_chi2,
                lower_percentile=lower_percentile, upper_percentile=upper_percentile,
                actual_peaks_to_fit=actual_peaks_to_fit
            )
            
            # --- NEW: Extract first 4 mu values and merge with metadata ---
            run_summary = metadata.copy()
            run_summary["file_name"] = Path(pq_filepath).name
            
            # Explicitly extract the requested metadata columns (defaults to None if missing)
            run_summary["temp_c"] = metadata.get("temp_c")
            run_summary["time_ns"] = metadata.get("time_ns")
            run_summary["bias_voltage"] = metadata.get("bias_voltage")
            run_summary["led_voltage"] = metadata.get("led_voltage")
            
            # Pre-fill with NaNs in case the fit failed
            for i in range(4):
                run_summary[f"mu_{i}"] = np.nan
                
            if fit_results.get("fit_successful") and len(fit_results.get("peak_params", [])) > 0:
                params = fit_results["peak_params"]
                # Safely grab up to 4 mus
                num_to_extract = min(4, len(params))
                for i in range(num_to_extract):
                    run_summary[f"mu_{i}"] = float(params[i, 0]) # Index 0 is mu, Index 1 is sigma
            
            summary_data.append(run_summary)
            # -------------------------------------------------------------

            # Subsample if necessary
            MAX_POINTS = 1_000_000
            plot_df = df_chan.sample(n=MAX_POINTS, random_state=42) if len(df_chan) > MAX_POINTS else df_chan
            
            areas = plot_df['peak_integral'].values
            baselines = plot_df['baseline'].values
            sigmas = plot_df['std_dev'].values
            
            # --- FILTER OUTLIERS FOR HISTOGRAMS ---
            filtered_baselines = filter_extreme_outliers(baselines, multiplier=5.0)
            filtered_sigmas = filter_extreme_outliers(sigmas, multiplier=5.0)
            
            fig, axes = plt.subplots(nrows=11, ncols=1, figsize=(7, 44), dpi=300)

            # Row 0: Peak Area Histogram
            plot_peak_area_histogram(axes[0], fit_results, colors[col_idx], metadata, target_channel, log)

            # Draw vertical lines for first 5 peaks
            if fit_results.get("fit_successful") and len(fit_results.get("peak_params", [])) > 0:
                first_mus = fit_results["peak_params"][:5, 0]
                for mu in first_mus:
                    axes[0].axvline(x=mu, color='black', linestyle=':', alpha=0.7)
                
                mu_text = "\n".join([f"μ{idx}: {mu:.0f}" for idx, mu in enumerate(first_mus)])
                axes[0].text(0.98, 0.80, mu_text, transform=axes[0].transAxes, va='top', ha='right', 
                             fontsize=9, bbox=dict(facecolor='white', alpha=0.9, edgecolor='lightgray', pad=4))
            
            # Rows 1 & 2: Gaussian fit histograms (using filtered data)
            fit_and_plot_gaussian_hist(axes[1], filtered_baselines, "Baseline Histogram", "Baseline", 'green')
            fit_and_plot_gaussian_hist(axes[2], filtered_sigmas, "Sigma Histogram", "Sigma", 'orange')

            # --- NEW: 3-Sigma Filter for Sample Signals ---
            mean_bl, std_bl = df_chan['baseline'].mean(), df_chan['baseline'].std()
            mean_sd, std_sd = df_chan['std_dev'].mean(), df_chan['std_dev'].std()
            
            valid_mask = (
                (np.abs(df_chan['baseline'] - mean_bl) <= 3 * std_bl) &
                (np.abs(df_chan['std_dev'] - mean_sd) <= 3 * std_sd)
            )
            
            valid_event_indices = df_chan[valid_mask].index.to_list()
            
            if len(valid_event_indices) < num_samples:
                print("Warning: Not enough events passed the 3-sigma filter. Falling back to all indices.")
                valid_event_indices = df_chan.index.to_list()

            # Rows 3-7: Sample Signals
            plot_signals(axes[3:8], root_filepath, target_channel, num_samples, device, valid_indices=valid_event_indices)

            # Rows 8-10: Hexbins
            plot_hexbin(axes[8], areas, baselines, "Baseline vs Area", "Area", "Baseline", fig)
            plot_hexbin(axes[9], areas, sigmas, "Sigma vs Area", "Area", "Sigma", fig)
            plot_hexbin(axes[10], sigmas, baselines, "Baseline vs Sigma", "Sigma", "Baseline", fig)

            plt.tight_layout()
            
            # Save temp image
            temp_img_path = temp_dir_path / f"col_{col_idx:03d}.png"
            fig.savefig(temp_img_path, format='png', bbox_inches='tight')
            temp_image_paths.append(temp_img_path)

            # Aggressive Teardown
            plt.close(fig)
            plt.close('all')
            del df_chan, plot_df, areas, baselines, sigmas, filtered_baselines, filtered_sigmas
            gc.collect()

        # Stitching Phase
        if not temp_image_paths:
            print(f"\nERROR: Target channel '{target_channel}' was not found in ANY files.")
            return 
            
        print("\n>>> All files processed. Stitching individual plots into a gigantic canvas... <<<")
        stitch_images_to_pdf(temp_image_paths, final_save_path)
        print(f"SUCCESS! Gigantic canvas PDF saved to: {final_save_path}")
        
        # --- Save the summary DataFrame to Parquet ---
        if summary_data and parquet_output_folder:
            summary_df = pd.DataFrame(summary_data)
            parquet_output_folder = Path(parquet_output_folder)
            parquet_output_folder.mkdir(parents=True, exist_ok=True)
            parquet_path = parquet_output_folder / f"{target_channel}_mu_summary.parquet"
            
            target_meta_cols = ["temp_c", "time_ns", "bias_voltage", "led_voltage"]
            
            cols = summary_df.columns.tolist()
            mu_cols = [c for c in cols if c.startswith('mu_')]
            other_cols = [c for c in cols if c not in mu_cols and c not in target_meta_cols and c != "file_name"]
            
            summary_df = summary_df[["file_name"] + target_meta_cols + mu_cols + other_cols]
            summary_df.to_parquet(parquet_path, index=False)
            print(f"SUCCESS! Mu parameters and metadata saved to: {parquet_path}")








from pathlib import Path

def compare_parquet_data_all_channels(
        parquet_folder, root_folder, 
        bins=500, 
        num_peaks=18,
        actual_peaks_to_fit=7,
        upper_percentile=95,
        lower_percentile=0, 
        num_samples=5,
        max_reduced_chi2=None, 
        target_channel=None, 
        plot_output_folder=None,
        parquet_output_folder=None, 
        log=False):
    
    # 1. Get a list of all parquet files in your parquet folder
    parquet_files_list = get_parquet_files(parquet_folder)

    # 2. Use the helper function to find the corresponding root files
    matched_roots = get_matching_root_files(parquet_files_list, root_folder)

    # 3. Filter out any parquet files that don't have a matching root file
    valid_pqs = []
    valid_rts = []
    for pq, rt in zip(parquet_files_list, matched_roots):
        if rt is not None:
            valid_pqs.append(pq)
            valid_rts.append(rt)

    # Determine the directory where PDFs are saved to check for existence
    actual_plot_folder = plot_output_folder if plot_output_folder is not None else "comparisons_folder"
    save_dir = Path(actual_plot_folder)

    # 4. NOW pass the matched lists into the plotting function
    if target_channel == None:
        target_channels = ['ch00','ch01','ch02','ch03','ch04','ch05','ch06','ch07','ch08','ch09','ch10','ch11','ch12','ch13','ch14','ch15']

        for channel in target_channels:
            # ==========================================
            # NEW: Skip computation if the PDF exists
            # ==========================================
            final_save_path = save_dir / f"{channel}_gigantic_canvas.pdf"
            if final_save_path.exists():
                print(f"Skipping channel '{channel}': {final_save_path} already exists.")
                continue
            # ==========================================
                
            print(f"Processing channel: {channel}")
            compare_parquet_and_root_data(
                parquet_files=valid_pqs,
                root_files=valid_rts,
                target_channel=channel,
                plot_output_folder=plot_output_folder,
                parquet_output_folder=parquet_output_folder,
                upper_percentile=upper_percentile,
                lower_percentile=lower_percentile,
                num_peaks=num_peaks,
                num_samples=num_samples,
                bins=bins,
                max_reduced_chi2=max_reduced_chi2,
                actual_peaks_to_fit=actual_peaks_to_fit,
                log=log
            )    
    else:
        # ==========================================
        # NEW: Skip computation if the PDF exists
        # ==========================================
        final_save_path = save_dir / f"{target_channel}_gigantic_canvas.pdf"
        if final_save_path.exists():
            print(f"Skipping channel '{target_channel}': {final_save_path} already exists.")
            return
        # ==========================================

        print(f"Processing channel: {target_channel}")
        compare_parquet_and_root_data(
            parquet_files=valid_pqs,
            root_files=valid_rts,
            target_channel=target_channel,
            plot_output_folder=plot_output_folder,
            parquet_output_folder=parquet_output_folder,
            upper_percentile=upper_percentile,
            lower_percentile=lower_percentile,
            num_peaks=num_peaks,
            num_samples=num_samples,
            bins=bins,
            max_reduced_chi2=max_reduced_chi2,
            actual_peaks_to_fit=actual_peaks_to_fit,
            log=log
        )








def get_gains(summary_files, target_files, plot_folder):
    # Ensure summary_files is a list, even if a single string is passed
    if isinstance(summary_files, str):
        summary_files = [summary_files]
        
    # Parse metadata of all target files once
    # (Assuming parse_folder_metadata is defined elsewhere)
    metadata = [parse_folder_metadata(f) for f in target_files]

    # Set up the plot
    plt.figure(figsize=(12, 6))

    # Dictionary to store results, now nested by channel
    results = {}

    # Loop over each summary file (representing different channels)
    for summary_file in summary_files:
        # Load the data
        df = pd.read_parquet(summary_file)
        
        # Extract channel from filename (e.g., looks for 'ch02', 'ch11', case-insensitive)
        base_name = os.path.basename(summary_file)
        ch_match = re.search(r'(ch\d+)', base_name, re.IGNORECASE)
        channel_name = ch_match.group(1).lower() if ch_match else base_name.split('.')[0]
        
        # Initialize the nested dictionary for this channel
        results[channel_name] = {}

        # Identify and sort the 'mu_' columns numerically
        mu_cols = [col for col in df.columns if str(col).startswith('mu_')]
        mu_cols = sorted(mu_cols, key=lambda x: int(x.split('_')[1]))
        x_values = np.arange(len(mu_cols))

        # Loop through files and metadata simultaneously
        for file_name, meta in zip(target_files, metadata):
            filtered_df = df[df['file_name'] == file_name]
            
            if not filtered_df.empty:
                y_values = filtered_df[mu_cols].iloc[0].values.astype(float)
                
                # Filter out NaN values before fitting
                valid_idx = np.isfinite(y_values)
                x_valid = x_values[valid_idx]
                y_valid = y_values[valid_idx]
                
                if len(x_valid) > 1:
                    # Perform linear fit
                    slope, intercept = np.polyfit(x_valid, y_valid, 1)
                    
                    # Store gain and mu values under the specific channel
                    results[channel_name][file_name] = {
                        'gain': slope,
                        'mu_values': dict(zip(mu_cols, y_values))
                    }
                    
                    fit_line = slope * x_valid + intercept
                    
                    # Create labels including the channel name
                    bias = meta.get('bias_voltage', 'N/A')
                    led = meta.get('led_voltage', 'N/A')
                    clean_label = f"{channel_name} | Bias: {bias}V, LED: {led}V"
                    
                    # Plot line and scatter, sharing the same color per pair
                    line, = plt.plot(x_valid, fit_line, linestyle='--', 
                                    label=f"[{channel_name}] Fit Gain: {slope:.1f}")
                    
                    plt.scatter(x_valid, y_valid, marker='o', color=line.get_color(), 
                                label=f"Data: {clean_label}")
                else:
                    print(f"Warning: Not enough valid data points to fit '{file_name}' for {channel_name}.")
            else:
                print(f"Warning: No data found for '{file_name}' in {channel_name} summary.")

    # Format the plot
    
    plt.xlabel('mu Index')
    plt.ylabel('Value')
    all_channels = ", ".join(results.keys())
    plt.title(f'mu Values and Linear Fits for channels: {all_channels}')
    
    # We use the last mu_cols to set ticks, assuming all files share the same mu_cols structure
    plt.xticks(ticks=x_values, labels=mu_cols, rotation=45)

    # Place legend outside
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the plot
    os.makedirs(plot_folder, exist_ok=True)
    plot_path = os.path.join(plot_folder, "gains_plot.pdf")
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"\nPlot successfully saved to: {plot_path}")
    plt.close()

    # Print summary and return
    print("\nExtracted Gains:")
    for ch, files_data in results.items():
        print(f"\n--- {ch.upper()} ---")
        for file, data in files_data.items():
            print(f"{file}: {data['gain']:.4f}")
            
    return results

    

def filter_extreme_outliers(data, multiplier=5.0):
    """
    Uses IQR to filter only extreme outliers. 
    A multiplier of 5.0 is very generous and will leave clean distributions completely untouched 
    while catching the crazy values stretching your axes.
    """
    q25, q75 = np.percentile(data, [25.0, 75.0])
    iqr = q75 - q25
    lower_bound = q25 - (multiplier * iqr)
    upper_bound = q75 + (multiplier * iqr)
    
    return data[(data >= lower_bound) & (data <= upper_bound)]























def _extract_params(name_string):
    """
    Helper function to apply all regex searches to a given string.
    Automatically handles both Type A (hyphenated) and Type B (underscored) formats.
    """
    metadata = {}
    
    # Lowercase everything for easier regex matching (handles 'V' vs 'v', 'C' vs 'c')
    ns = name_string.lower()
    
    # Auto-detect file format: Type B uses underscores, Type A uses only hyphens
    is_type_b = "_" in ns
    
    # 1. Extract Timestamp (Typically Type A only)
    timestamp_match = re.match(r"^(\d{8}t\d{6}z)", ns)
    if timestamp_match:
        metadata["timestamp"] = timestamp_match.group(1).upper()
        
    # 2 & 11. Extract Channel Type / Mode
    mode_match = re.search(r"(se[-_]multi|multi|single|sgl)", ns)
    if mode_match:
        mode_str = mode_match.group(1).replace('_', '-').upper()
        if mode_str == "SGL":
            mode_str = "SINGLE"
        metadata["mode"] = mode_str
        metadata["channel_type"] = mode_str.lower() + "-ch"

    # 3 & 8. Extract Voltages (Bias and LED/Pulse)
    # Looks for matches like "-54v", "_54v", "-2-89v", "_3.1v"
    all_voltages = re.findall(r"[-_]([0-9]+(?:[-.][0-9]+)?)[v]", ns)
    if all_voltages:
        # Helper to parse float based on the file format
        def parse_v(v_str):
            return float(v_str) if is_type_b else float(v_str.replace("-", "."))
            
        metadata["bias_voltage"] = parse_v(all_voltages[0])
        
        if len(all_voltages) >= 2:
            led_v = parse_v(all_voltages[-1])
            metadata["pulse_voltage"] = led_v
            metadata["led_voltage"] = led_v

    # 4. Extract Frequency (handles hz and khz)
    freq_match = re.search(r"[-_](\d+)(k?)hz", ns)
    if freq_match:
        val = float(freq_match.group(1))
        is_kilo = freq_match.group(2) == 'k'
        metadata["frequency_hz"] = val * 1000 if is_kilo else val

    # 5. Extract Time window/pulse width
    time_match = re.search(r"[-_](\d+)ns", ns)
    if time_match:
        metadata["time_ns"] = float(time_match.group(1))

    # 6. Extract Temp/Condition
    temp_match = re.search(r"[-_](-?\d+)c(?:[-_]|$)", ns)
    if temp_match:
        temp_val = float(temp_match.group(1))
        # Type A stores "160" but implies -160. Type B explicitly stores "-120".
        if temp_val > 40:
            temp_val = -temp_val
        metadata["temp_c"] = temp_val

    # 7. Extract the unique trailing hash (Type A only)
    hash_match = re.search(r"-([a-f0-9]{8})$", ns)
    if hash_match:
        metadata["run_hash"] = hash_match.group(1)

    # 9. Check for dark_count
    if re.search(r"dark[-_]?count", ns):
        metadata["is_dark_count"] = True

    # 10. Extract Amplifier Range 
    # Finds the two sets of numbers between the 'c' (temp) and the 'v' (bias voltage)
    amp_match = re.search(r"c[-_]([+-]?\d+(?:[-.]\d+)?)[-_]([+-]?\d+(?:[-.]\d+)?)[-_](?=\d+(?:[-.]\d+)?[v])", ns)
    
    if amp_match:
        val1_str = amp_match.group(1)
        val2_str = amp_match.group(2)
        
        if is_type_b:
            # Type B has explicit signs and explicit decimal dots (e.g., "+0.5" and "-3")
            val1, val2 = float(val1_str), float(val2_str)
            metadata["amp_min_v"] = min(val1, val2)
            metadata["amp_max_v"] = max(val1, val2)
        else:
            # Type A uses hyphens for decimals (e.g., "0-5" and "2-5") and implicit negative min
            val1 = float(val1_str.replace("-", "."))
            val2 = float(val2_str.replace("-", "."))
            metadata["amp_min_v"] = -val1
            metadata["amp_max_v"] = val2

    return metadata


def parse_folder_metadata(filepath):
    """
    Extracts experimental parameters from the parent folder's name, 
    falling back to the file name if a parameter is missing.
    """
    folder_name = os.path.basename(os.path.dirname(filepath))
    file_name = os.path.splitext(os.path.basename(filepath))[0]
    
    file_meta = _extract_params(file_name)
    folder_meta = _extract_params(folder_name)
    
    # **file_meta sets the defaults, **folder_meta overwrites them.
    metadata = {
        "raw_folder_name": folder_name,
        **file_meta,
        **folder_meta 
    }
    
    return metadata














####################################################################
# CSV DATA
####################################################################

#Inputs csv generated from Davids Oscilloscope, with the first two columns being time and signal.
def compute_photon_signal_single_peak(filepath):
    """
    Looks for the photon peak in the signal, extracts it and integrates the
    signal above a 3 sigma threshold.

    Parameters:
    filepath (str): The path to the file containing the photon signal data.
                    A csv file which has x and y values on the first two columns.

    Returns:
    float, float, float: area, sigma, bkg average value
    """
    ###############################################################################################
    # Actual implementation
    ###############################################################################################

    df = pd.read_csv(filepath)
    df = df.rename(columns={'in s': 'time_s', 'C1 in V': 'signal_v'})

    peak = df['signal_v'].max()
    peak_time = df.loc[df['signal_v'] == peak, 'time_s'].iloc[0]
    total_avg = df['signal_v'].mean()  # abritrarly selected baseline

    # search the point where the signal reaches the total_avg after the maximum peak
    post_peak_df = df[(df['time_s'] >= peak_time) & (df['signal_v'] <= total_avg)]
    if not post_peak_df.empty:
        # get the first point after the peak that drops to baseline
        peak_end = post_peak_df.iloc[0]

    else:
    #don't save anything and skip it entirely
        return np.nan


    # do the same for the pre peak point earlier in time
    pre_peak_df = df[(df['time_s'] <= peak_time) & (df['signal_v'] <= total_avg)]

    if not pre_peak_df.empty:
        # get the last point before the peak that was at baseline
        peak_start = pre_peak_df.iloc[-1]

    else:
    #don't save anything and skip it entirely
        return np.nan



    # create a mask that removes the signal
    pre_peak_bkg = (df['time_s'] < peak_start['time_s'])
    post_peak_bkg = (df['time_s'] > peak_end['time_s'])

    # call it background
    background_df = df[pre_peak_bkg]
    background_df = background_df.rename(columns={'signal_v': 'pre_peak_bkg_v'})

    # save the rest of the data as signal
    signal_df = df[~(post_peak_bkg & pre_peak_bkg)]

    # average the background signal

    background_avg = background_df['pre_peak_bkg_v'].mean()

    # compute one sigma deviation of the background_avg
    sigma = background_df['pre_peak_bkg_v'].std()

    significant_signals = (signal_df['signal_v'] > background_avg + 3 * sigma)
    signal_above_sigma = signal_df[significant_signals]
    area = np.trapezoid(signal_above_sigma['signal_v'] - (background_avg + 3 * sigma), signal_above_sigma['time_s'])

    return area, sigma, background_avg









#computes area of peak, baseline level and sigma of baseline
#input is the file obtained from the 4ch desktop digitizer
def compute_DAQ_data(filepath):
    """
    Extracts photon signals.
    
    Parameters:
    filepath (str): Path to the CSV file
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        original_keys = f.readline().strip().split(';')
        if 'SAMPLES' in original_keys:
            original_keys[original_keys.index('SAMPLES')] = 'SAMPLES1'
        max_cols = max(len(line.strip().split(';')) for line in f)

    if max_cols > len(original_keys):
        extra_keys_needed = max_cols - len(original_keys)
        for i in range(extra_keys_needed):
            original_keys.append(f"SAMPLES{i+2}")

    df = pd.read_csv(filepath, sep=';', names=original_keys, skiprows=1, index_col=False)
    
    
    samples_matrix = df.filter(like='SAMPLES').to_numpy()
    master_time = np.arange(samples_matrix.shape[1])

    array_area = []
    array_sigma = []
    array_bkg = []


    total_events = 0



    for row_index, signal in enumerate(samples_matrix):
        total_events += 1 #delete afterwards

        valid_mask = ~np.isnan(signal)
        analysis_sig = signal[valid_mask]
        t = master_time[valid_mask]
        
        if len(analysis_sig) < 50: 
            continue

        # 2. Calculate True Baseline (Assuming first 10% of waveform is pre-trigger)
        bkg_end = max(10, int(len(analysis_sig) * 0.1))
        bkg_sig = analysis_sig[:bkg_end]
        
        baseline = np.mean(bkg_sig)
        baseline_sigma = np.std(bkg_sig, ddof=1) if len(bkg_sig) > 1 else 0.0

        
        # 1-Photon Event
        peak_idx = len(t)//2+43
        
        # A. Calculate a LOCAL baseline right before the peak 
        # This compensates for the slow, wavy drift in your raw data
        bkg_start = 0
        bkg_end = max(1, peak_idx - 35)
        local_bkg_sig = analysis_sig[bkg_start:bkg_end]
        
        if len(local_bkg_sig) > 0:
            local_baseline = np.mean(local_bkg_sig)
        else:
            local_baseline = baseline

        # B. Find Pre-Peak Intersection 35 samples before peak
        pre_peak_idx = max(0, peak_idx - 35)

        # C. Find Post-Peak Intersection 60 samples after peak
        post_peak_idx = peak_idx + 60

        # 5. Extract and Integrate 
        signal_window = analysis_sig[pre_peak_idx : post_peak_idx]
        time_window = t[pre_peak_idx : post_peak_idx]

        if len(signal_window) < 2:
            continue

        # Subtract the local baseline for accurate charge calculation
        area = np.trapezoid(signal_window - local_baseline, time_window)

        array_area.append(area)
        array_sigma.append(baseline_sigma)
        array_bkg.append(local_baseline)


    print(f"Total events: {total_events}")
    print(f"Successfully integrated: {len(array_area)}")

    return array_area, array_sigma, array_bkg



#processes files in chunks
#data obtained from the big digitizer (64ch)

def process_csv_file_in_chunks_pandas(filepath,parquet_name, chunk_size=1000):
    # Optional: keep track of baselines if you want to plot them later
    integrated_areas  = []
    baseline_averages = []
    sigma_averages    = []
    channel_ids        = []

    peak_indx  = 580
    start_indx = 560
    end_indx   = 650

    for chunk in pd.read_csv(filepath, chunksize=chunk_size):
        
        # 1. Grab the event ID and channel id 
        event_id = chunk['event'].iloc[0]
        channel_id = chunk['channel'].iloc[0]
        channel_ids.append(channel_id)

        if chunk['event'].nunique() > 1:
            raise ValueError(f"Chunk contains multiple events! Chunk started with event {event_id}.")
        
        # 2. Filter the chunk for just the baseline samples (0 to 560)
        baseline_region = chunk[(chunk['sample'] >= 0) & (chunk['sample'] <= start_indx)]
        
        # 3. Compute the mean of the 'adc' column for this specific event
        event_baseline = baseline_region['adc'].mean()
        baseline_averages.append(event_baseline)

        # 4. Compute the standard deviation (sigma) of the CURRENT event's baseline
        sigma = baseline_region['adc'].std()
        sigma_averages.append(sigma)

        # 5. Find the signal window
        signal_window = chunk[(chunk['sample'] >= start_indx) & (chunk['sample'] <= end_indx)]

        # 6. Calculate the deviation from the baseline
        deviation = signal_window['adc'] - event_baseline

        # 7. Prepare values for integration
        # np.abs() handles both negative and positive peaks by looking at magnitude.
        # If the magnitude of the pulse is > 3*sigma, keep it; otherwise set to 0.
        y_vals = np.abs(deviation)
        x_vals = signal_window['sample']

        # 8. Integrate the signal in this region
        area = np.trapezoid(y_vals, x=x_vals)
        integrated_areas.append(area)   

    result_df = pd.DataFrame({
        'channel': channel_ids,
        'areas'  : integrated_areas,
        'sigmas' : sigma_averages,
        'baseline': baseline_averages
    }) 

    output_folder = f'/disk/gfs_atp/hekinc/Master_measurements/{parquet_name}.parquet'      

    result_df.to_parquet(output_folder,index=False)
    return result_df












def plotter(array_area, array_sigma, array_bkg, bins=500):
    """
    Plots the results of the photon signal analysis.

    Parameters:
    array_area (list): List of integrated areas for each event.
    array_sigma (list): List of baseline noise (sigma) for each event.
    array_bkg (list): List of baseline averages for each event.
    """

    # 1. Filter out any NaNs from all arrays just to be safe
    clean_area = [a for a in array_area if not np.isnan(a)]
    clean_sigma = [s for s in array_sigma if not np.isnan(s)]
    clean_bkg = [b for b in array_bkg if not np.isnan(b)]

    # 2. Automatically calculate the "core" range for Sigma and Bkg 
    # This finds the boundaries that contain 99% of your data, ignoring massive outliers
    sigma_min, sigma_max = np.percentile(clean_sigma, [0, 99])
    bkg_min, bkg_max = np.percentile(clean_bkg, [1, 99])

    # 3. Create a figure with 3 subplots in a single row
    fig, axs = plt.subplots(2, 3, figsize=(18, 8))

    # --- Plot 1: Area (Charge Spectrum) ---
    axs[0, 0].hist(clean_area, bins=bins, color='blue', alpha=0.7)
    axs[0, 0].set_title('Histogram of Area (Core)')
    axs[0, 0].set_xlabel('Area')
    axs[0, 0].set_ylabel('Frequency')
    axs[0, 0].grid(True, alpha=0.3)

    # --- Plot 2: Sigma (Baseline Noise) ---
    axs[0, 1].hist(clean_sigma, bins=200, range=(sigma_min, sigma_max), color='green', alpha=0.7)
    axs[0, 1].set_title('Histogram of Sigma (Noise)')
    axs[0, 1].set_xlabel('Sigma')
    axs[0, 1].set_ylabel('Frequency')
    axs[0, 1].grid(True, alpha=0.3)

    # --- Plot 3: Background Baseline ---
    axs[0, 2].hist(clean_bkg, bins=200, range=(bkg_min, bkg_max), color='red', alpha=0.7)
    axs[0, 2].set_title('Histogram of Background Baseline')
    axs[0, 2].set_xlabel('Baseline Level')
    axs[0, 2].set_ylabel('Frequency')
    axs[0, 2].grid(True, alpha=0.3)


    axs[1,0].plot(clean_area, clean_sigma, 'o', markersize=2, alpha=0.5)
    axs[1,0].set_title('Sigma vs Area')
    axs[1,0].set_xlabel('Area')
    axs[1,0].set_ylabel('Sigma')
    axs[1,0].grid(True, alpha=0.3)

    axs[1,1].plot(clean_area, clean_bkg, 'o', markersize=2, alpha=0.5)
    axs[1,1].set_title('Background Baseline vs Area')
    axs[1,1].set_xlabel('Area')
    axs[1,1].set_ylabel('Background Baseline')
    axs[1,1].grid(True, alpha=0.3)

    axs[1,2].plot(clean_sigma, clean_bkg, 'o', markersize=2, alpha=0.5)
    axs[1,2].set_title('Background Baseline vs Sigma')
    axs[1,2].set_xlabel('Sigma')
    axs[1,2].set_ylabel('Background Baseline')
    axs[1,2].grid(True, alpha=0.3)

    # Adjust spacing so labels don't overlap and show the plots
    plt.tight_layout()
    plt.show()











# --------------------------------------------------------- 
# 2. Main Analysis and Fitting Function 
# --------------------------------------------------------- 

def analyze_and_fit_physics(raw_data, bins=500, num_peaks_to_fit=18, 
                            confident_peak_limit=7, manual_mu0=None, manual_gain=None): 
    """ 
    Fits physical data using a constrained multi-photon physics model. 
    If manual_mu0 and manual_gain are provided, the automatic peak detector is bypassed. 
    """ 
    raw_data = np.array(raw_data, dtype=float) 
    print(f"Processing {len(raw_data)} data points...") 

    counts, bin_edges = np.histogram(raw_data, bins=bins) 
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 
    dx = bin_centers[1] - bin_centers[0] 

    # ========================================== 
    # Step A & B: Find Anchors (Manual or Auto) 
    # ========================================== 
    if manual_mu0 is not None and manual_gain is not None: 
        print("Using manual physics overrides for Pedestal and Gain.") 
        mu_0_guess = manual_mu0 
        gain_guess = manual_gain 
    else: 
        print("Auto-detecting peaks...") 
        smoothed_counts = gaussian_filter1d(counts, sigma=1) 
        peaks, _ = find_peaks( 
            smoothed_counts,  
            prominence=np.max(smoothed_counts) * 0.01,  
            distance=4 
        ) 
        
        # Fail-safe to ensure we at least capture the pedestal 
        global_max_bin = np.argmax(counts) 
        if not any(abs(p - global_max_bin) <= 5 for p in peaks): 
            peaks = np.append(peaks, global_max_bin) 
            peaks = np.sort(peaks) 

        if len(peaks) == 0: 
            print("Failed to find any peaks. Try providing manual_mu0 and manual_gain.") 
            return None, None 

        idx_0 = peaks[0] 
        mu_0_guess = bin_centers[idx_0] 
        
        # Estimate gain (distance to next peak). If no 2nd peak, guess based on range. 
        if len(peaks) > 1: 
            gain_guess = bin_centers[peaks[1]] - bin_centers[peaks[0]] 
        else: 
            gain_guess = (bin_centers[-1] - bin_centers[0]) / (num_peaks_to_fit / 2) 

    # Estimate widths 
    sigma_0_guess = max(dx, 1e-9) 
    sigma_1_guess = sigma_0_guess * 0.5  

    # ========================================== 
    # Step C: Build Physics-Based Guesses & Bounds 
    # ========================================== 
    p0 = [] 
    lower_bounds = [] 
    upper_bounds = [] 

    # 1. Background Guesses 
    A_bg_guess = np.max(counts) * 0.1 
    lambda_bg_guess = 1.0 / max(np.mean(raw_data), 1e-9) 
    p0.extend([A_bg_guess, lambda_bg_guess]) 
    
    lower_bounds.extend([0, 0]) 
    upper_bounds.extend([np.max(counts) * 0.5, 5000]) # Strict background ceiling 

    # 2. Physics Parameter Guesses (mu_0, sigma_0, gain, sigma_1) 
    p0.extend([mu_0_guess, sigma_0_guess, gain_guess, sigma_1_guess]) 
    
    # BOUNDS: Force sigma_0 and sigma_1 to stay under 40% of the gain, preserving the valleys! 
    lower_bounds.extend([bin_centers[0] - dx*10, 1e-12, dx, 1e-12]) 
    upper_bounds.extend([bin_centers[-1], gain_guess * 1.2, gain_guess * 5, gain_guess * 1.2]) 

    # 3. Amplitude Guesses for ALL peaks (0 through num_peaks_to_fit) 
    for n in range(num_peaks_to_fit): 
        expected_mu = mu_0_guess + (n * gain_guess) 
        
        # Guess the amplitude by looking at the histogram height at that x-value 
        if bin_centers[0] <= expected_mu <= bin_centers[-1]: 
            closest_bin_idx = np.abs(bin_centers - expected_mu).argmin() 
            A_guess = counts[closest_bin_idx] 
        else: 
            A_guess = 1e-9  
            
        p0.append(A_guess) 
        lower_bounds.append(0.0) 
        upper_bounds.append(np.inf) 

    # Force guesses strictly inside bounds 
    p0 = np.clip(np.array(p0, dtype=float), np.array(lower_bounds, dtype=float) + 1e-10, np.array(upper_bounds, dtype=float) - 1e-10) 

    # ========================================== 
    # Step D: Fit the curve 
    # ========================================== 
    try: 
        popt, pcov = curve_fit( 
            fit_model_physics,  
            bin_centers,  
            counts,  
            p0=p0,  
            bounds=(lower_bounds, upper_bounds), 
            maxfev=25000  
        ) 
        print("Physics Curve fitting converged successfully!") 

        expected_counts = fit_model_physics(bin_centers, *popt)
            
        # Only calculate chi-square for bins with at least 1 count to avoid dividing by zero
        valid_bins = counts > 0
        O_i = counts[valid_bins]
        E_i = expected_counts[valid_bins]
        
        # Calculate Chi-Square
        chi_square = np.sum(((O_i - E_i) ** 2) / O_i)
        
        # Calculate Degrees of Freedom (Number of data points - Number of fitted parameters)
        ndf = len(O_i) - len(popt)
        
        # Calculate Reduced Chi-Square
        reduced_chi_square = chi_square / ndf if ndf > 0 else np.nan
        # -----------------------------------

    except Exception as e: 
        print(f"Optimal parameters not found: {e}") 
        return None, None 

    # ========================================== 
    # Step E: Plotting and Output 
    # ========================================== 
    plt.figure(figsize=(10, 6)) 
    plt.hist(raw_data, bins=bins, alpha=0.5, color='blue', label='Binned Data') 
    x_fit = np.linspace(bin_centers[0], bin_centers[-1], 1000) 
    
    # Plot Total Fit 
    plt.plot(x_fit, fit_model_physics(x_fit, *popt), color='red', lw=2, label='Total Fit') 
    
    # Extract optimized physics parameters 
    A_bg, lambda_bg, mu_0, sigma_0, gain, sigma_1 = popt[0:6] 
    amplitudes = popt[6:] 
    
    # Plot Exponential Background 
    #plt.plot(x_fit, A_bg * np.exp(-lambda_bg * x_fit), color='black', linestyle=':', lw=2, label='Exp Background') 
    
    print("\n--- Physical Fit Results ---") 
    print(f"Pedestal (\u03BC_0): {mu_0:.6e}") 
    print(f"Detector Gain:      {gain:.6e}") 
    print(f"Pedestal Noise (\u03C3_0): {sigma_0:.6e}") 
    print(f"1-Photon Noise (\u03C3_1): {sigma_1:.6e}") 
    print(f"\u03C7\u00B2 / NDF:           {chi_square:.2f} / {ndf} = {reduced_chi_square:.3f}")
    print("-" * 28) 
    
    tail_sum = np.zeros_like(x_fit) 
    
    for n, A in enumerate(amplitudes): 
        mu_n = mu_0 + n * gain 
        sigma_n = np.sqrt(sigma_0**2 + n * sigma_1**2) 
        single_gauss = A * np.exp(-0.5 * ((x_fit - mu_n) / sigma_n)**2) 
        
        # Only plot and print individual peaks up to the limit 
        if n <= confident_peak_limit: 
            # Check if amplitude is large enough to bother plotting 
            if np.max(single_gauss) > (np.max(counts) * 0.005): 
                plt.plot(x_fit, single_gauss, '--', label=f'{n}-Photon') 
                print(f"Peak {n}: \u03BC = {mu_n:.6e}, \u03C3 = {sigma_n:.6e}") 
        else: 
            # Group the higher-order peaks into the tail 
            tail_sum += single_gauss 

    # Plot the aggregated High-Photon Tail if it exists 
    if np.max(tail_sum) > (np.max(counts) * 0.005): 
        plt.plot(x_fit, tail_sum, color='gray', linestyle='-.', lw=1.5, label='High-Photon Tail') 

    plt.xlabel('ADC Counts') 
    plt.ylabel('Frequency') 
    plt.title('Histogramm of Area') 
    
    # Move legend outside the plot if it gets too crowded 
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') 
    plt.grid(alpha=0.3) 
    plt.tight_layout() 
    plt.show() 

    return popt, pcov






#function that takes the filepath and frequency and returns 10 sample plots of the signal with the peak, baseline and sigma marked on the plot. The plots should be saved in a folder called "plots" in the current working directory.
def plot_sample_signals(filepath,conditions, num_samples=10, ):
    """
    Plots sample signals from the DAQ data with peak, baseline, and sigma marked.

    Parameters:
    filepath (str): Path to the CSV file.
    conditions (list): Experimental conditions [Temp, LED V, SiPM Bias V].
    num_samples (int): Number of sample plots to generate.
    """
   

    # Create a directory for plots if it doesn't exist
    #if not os.path.exists('plots'):
    #    os.makedirs('plots')

    with open(filepath, 'r', encoding='utf-8') as f:
        original_keys = f.readline().strip().split(';')
        #rename the first SAMPLES key to SAMPLES1 to avoid duplicate column names
        original_keys[original_keys.index('SAMPLES')] = 'SAMPLES1'
        
        # Scan the rest of the file to find the longest row
        max_cols = max(len(line.strip().split(';')) for line in f)

    # Generate new keys to fill the gap (e.g., if max_cols is 10 and original is 4)
    if max_cols > len(original_keys):
        extra_keys_needed = max_cols - len(original_keys)
        for i in range(extra_keys_needed):
            original_keys.append(f"SAMPLES{i+2}")

    # Pass 2: Let Pandas read the whole file using your newly extended keys
    df = pd.read_csv(
        filepath, 
        sep=';', 
        names=original_keys, 
        skiprows=1,           # Skip the original broken header
        index_col=False
    )

    # Extract data to numpy array
    samples_matrix = df.filter(like='SAMPLES').to_numpy()
    master_time = np.arange(samples_matrix.shape[1])

    # 1. Determine how many samples we can actually pick
    total_rows = samples_matrix.shape[0]
    actual_num_samples = min(num_samples, total_rows)

    # 2. Pick random row indices without replacement
    random_indices = np.random.choice(total_rows, size=actual_num_samples, replace=False)

    # 3. Iterate over the randomly selected indices
    for i, row_idx in enumerate(random_indices):
        signal = samples_matrix[row_idx]   # <-- Grab the randomly selected row
        valid_mask = ~np.isnan(signal)
        sig = signal[valid_mask]
        t = master_time[valid_mask]

        if len(sig) < 50: 
            continue

        # Calculate baseline and sigma
        bkg_end = max(10, int(len(sig) * 0.1))
        bkg_sig = sig[:bkg_end]
        baseline = np.mean(bkg_sig)
        sigma = np.std(bkg_sig, ddof=1) if len(bkg_sig) > 1 else 0.0

        # Find peak
        peak_idx = len(t)//2+43
        start_indx = peak_idx - 35
        end_indx = peak_idx + 60

        t_window = t[start_indx:end_indx]
        sig_window = sig[start_indx:end_indx]

        
        #peak_value = sig[peak_idx]

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(t, sig, label='Signal', color='blue')
        plt.axhline(y=baseline, color='green', linestyle='--', label='Baseline')
        plt.axhline(y=baseline +sigma, color='orange', linestyle='--', label='1 Sigma Threshold')
        plt.axvline(x=t[peak_idx], color='red', linestyle='--', label='Peak')
        plt.axvline(x=t[start_indx], color='black', linestyle='--', label='Peak-35 Samples')
        plt.axvline(x=t[end_indx], color='black', linestyle='--', label='Peak+60 Samples')

        #colour the area between three sigma line and the peak
        
        plt.fill_between(t_window, sig_window, baseline, 
                         where=(sig_window > baseline), 
                         color='purple', alpha=0.3, interpolate=True, label='Peak Area')
     
        
        # It's helpful to include the actual row index in the title and filename
        plt.title(
        f'Sample Signal {i+1} (Original Row: {row_idx})\n'
        f'Data taken at {float(conditions[0]):.1f} °C / {float(conditions[0])+273.15:.1f} K\n'
        f'LED: {float(conditions[1]):.1f} V | SiPM Bias: {float(conditions[2]):.1f} V'
        )
        plt.xlabel('Samples')

        plt.ylabel('Signal (ADC)')
        plt.legend()
        plt.grid()
        
        # Save the plot
        plt.show()



################################################################################
#ARCHIVES 
################################################################################
'''


'''