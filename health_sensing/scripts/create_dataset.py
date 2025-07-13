# ----------------------------
# scripts/create_dataset.py
# ----------------------------
import os
import pandas as pd
import numpy as np
import argparse
from scipy.signal import butter, filtfilt
from datetime import timedelta

WINDOW_SIZE = 30  #seconds
OVERLAP = 0.5
FS = 32


def bandpass_filter(signal, lowcut=0.17, highcut=0.4, fs=32, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, signal)


def load_custom_signal(file_lines):
    #Find the data start line 
    data_start = None
    for i, line in enumerate(file_lines):
        if line.strip().lower() in ["data:", "data"]:
            data_start = i
            break
    
    if data_start is None:
        raise ValueError("Could not find data section in file")
    
    data_lines = file_lines[data_start + 1:]

    timestamps = []
    values = []
    for line in data_lines:
        try:
            time_str, val = line.strip().split(';')
            timestamp = pd.to_datetime(time_str.strip(), format="%d.%m.%Y %H:%M:%S,%f")
            value = float(val.strip())
            timestamps.append(timestamp)
            values.append(value)
        except:
            continue

    return np.array(values), timestamps


def load_events(file_lines):
    events = []
    for line in file_lines:
        line = line.strip()
        if ";" in line and any(keyword in line for keyword in ["Hypopnea", "Obstructive Apnea", "Apnea"]):
            try:
                #given format: "30.05.2024 23:48:45,119-23:49:01,408; 16;Hypopnea; N1"
                parts = line.split(";")
                if len(parts) >= 3:
                    time_range = parts[0].strip()
                    label = parts[2].strip()
                    
                    #parse time range
                    if "-" in time_range:
                        start_str, end_str = time_range.split("-")
                        start_time = pd.to_datetime(start_str.strip(), format="%d.%m.%Y %H:%M:%S,%f")
                        end_time = pd.to_datetime(end_str.strip(), format="%H:%M:%S,%f")
                        
                        #Fix end_time to have same date as start_time
                        end_time = end_time.replace(year=start_time.year, month=start_time.month, day=start_time.day)
                        
                        # Handle day rollover
                        if end_time < start_time:
                            end_time = end_time + pd.Timedelta(days=1)
                        
                        events.append((start_time, end_time, label))
            except Exception as e:
                print(f"Failed to parse event line: {line} - Error: {e}")
                continue
    
    return pd.DataFrame(events, columns=["start_time", "end_time", "event"])


def find_file_by_pattern(folder, pattern):
    """Find file that contains the pattern in its name"""
    files = os.listdir(folder)
    for file in files:
        if pattern.lower() in file.lower():
            return file
    return None

def create_windows(data_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    all_windows = []
    participants = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]

    for pid in participants:
        folder = os.path.join(data_dir, pid)
        print(f"Processing participant {pid}...")

        # Flow file
        flow_file = None
        flow_patterns = ["flow -", "flow signal", "flow nasal"]
        for pattern in flow_patterns:
            potential_file = find_file_by_pattern(folder, pattern)
            if potential_file and "events" not in potential_file.lower():
                flow_file = potential_file
                break
        
        # Thoracic file  
        thorac_file = None
        thorac_patterns = ["thorac -", "thorac signal", "thorac movement"]
        for pattern in thorac_patterns:
            potential_file = find_file_by_pattern(folder, pattern)
            if potential_file:
                thorac_file = potential_file
                break
        
        # SPO2 file
        spo2_file = None
        spo2_patterns = ["spo2 -", "spo2 signal"]
        for pattern in spo2_patterns:
            potential_file = find_file_by_pattern(folder, pattern)
            if potential_file:
                spo2_file = potential_file
                break
        
        #events file
        events_file = find_file_by_pattern(folder, "flow events")
        
        if not all([flow_file, thorac_file, spo2_file, events_file]):
            print(f"Warning: Missing files for {pid}, skipping...")
            continue

        with open(os.path.join(folder, flow_file), 'r') as f:
            nasal, timestamps = load_custom_signal(f.readlines())
        with open(os.path.join(folder, thorac_file), 'r') as f:
            thoracic, _ = load_custom_signal(f.readlines())
        with open(os.path.join(folder, spo2_file), 'r') as f:
            spo2, _ = load_custom_signal(f.readlines())
        with open(os.path.join(folder, events_file), 'r') as f:
            events = load_events(f.readlines())

        #filter nasal and thoracic
        nasal = bandpass_filter(nasal)
        thoracic = bandpass_filter(thoracic)

        win_len = int(FS * WINDOW_SIZE)
        step = int(win_len * (1 - OVERLAP))
        total_len = len(nasal)

        for i in range(0, total_len - win_len, step):
            window_id = f"{pid}_{i}"
            win_start = timestamps[i]
            win_end = timestamps[i + win_len - 1]

            label = "Normal"
            for _, row in events.iterrows():
                overlap = max(0, (min(win_end, row.end_time) - max(win_start, row.start_time)).total_seconds())
                if overlap / WINDOW_SIZE > 0.5:
                    label = row.event
                    break

            win_dict = {
                'participant_id': pid,
                'window_id': window_id,
                'nasal': nasal[i:i+win_len].tolist(),
                'thoracic': thoracic[i:i+win_len].tolist(),
                'spo2': spo2[i//8:(i+win_len)//8].tolist(),
                'label': label
            }
            all_windows.append(win_dict)

    pd.DataFrame(all_windows).to_csv(os.path.join(out_dir, 'breathing_dataset.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-in_dir", type=str, required=True)
    parser.add_argument("-out_dir", type=str, required=True)
    args = parser.parse_args()
    create_windows(args.in_dir, args.out_dir)
