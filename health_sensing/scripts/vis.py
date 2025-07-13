import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import argparse
import os
from matplotlib.backends.backend_pdf import PdfPages

# Function to load a signal file
def load_signal(filepath, sample_rate):
    with open(filepath, 'r') as file:
        lines = file.readlines()
    start_index = lines.index('Data:\n') + 1
    data = [line.strip().split(';') for line in lines[start_index:] if line.strip()]
    df = pd.DataFrame(data, columns=['Time', 'Value'])
    df['Time'] = pd.to_datetime(df['Time'], format='%d.%m.%Y %H:%M:%S,%f')
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df.set_index('Time', inplace=True)
    return df

# Function to load event annotations
def load_events(file_path):
    # Read the file and skip header lines
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    #Find the start of data 
    data_lines = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith('Signal') and not line.startswith('Start Time') and not line.startswith('Unit'):
            if ';' in line and '-' in line:  
                data_lines.append(line)
    
    events = []
    for line in data_lines:
        try:
            parts = line.split(';')
            if len(parts) >= 3:
                time_range = parts[0].strip()
                duration = parts[1].strip()
                event_type = parts[2].strip()
                
                # Parse time range
                if '-' in time_range:
                    start_str, end_str = time_range.split('-')
                    start_time = pd.to_datetime(start_str.strip(), format='%d.%m.%Y %H:%M:%S,%f')
                    end_time = pd.to_datetime(f"{start_time.strftime('%d.%m.%Y')} {end_str.strip()}", format='%d.%m.%Y %H:%M:%S,%f')
                    
                    events.append({
                        'Start': start_time,
                        'End': end_time,
                        'Event': event_type
                    })
        except Exception as e:
            print(f"Warning: Could not parse line: {line} - {e}")
            continue
    
    return pd.DataFrame(events)

def find_file_by_pattern(folder, pattern):
    """Find file that contains the pattern in its name"""
    files = os.listdir(folder)
    for file in files:
        if pattern.lower() in file.lower():
            return file
    return None

# Main visualization function
def generate_visualization(folder_path):
    output_dir = os.path.join(os.path.dirname(folder_path), '..', 'Visualizations')
    os.makedirs(output_dir, exist_ok=True)
    
    participant_name = os.path.basename(folder_path)
    pdf_path = os.path.join(output_dir, f'{participant_name}_visualization.pdf')

    # Find files using pattern matching
    flow_file = find_file_by_pattern(folder_path, "flow") and not find_file_by_pattern(folder_path, "flow events")
    if not flow_file:
        flow_file = find_file_by_pattern(folder_path, "flow")
        if "events" in flow_file.lower():
            # Find the other flow file
            files = os.listdir(folder_path)
            for f in files:
                if "flow" in f.lower() and "events" not in f.lower():
                    flow_file = f
                    break
    
    thor_file = find_file_by_pattern(folder_path, "thorac")
    spo2_file = find_file_by_pattern(folder_path, "spo2")
    event_file = find_file_by_pattern(folder_path, "flow events")

    if not all([flow_file, thor_file, spo2_file, event_file]):
        print(f"Error: Missing files in {folder_path}")
        return

    # Load data
    flow_df = load_signal(os.path.join(folder_path, flow_file), 32)
    thor_df = load_signal(os.path.join(folder_path, thor_file), 32)
    spo2_df = load_signal(os.path.join(folder_path, spo2_file), 4)
    events_df = load_events(os.path.join(folder_path, event_file))

    # Time segmentation (e.g., 5 min windows)
    start_time = flow_df.index.min()
    end_time = flow_df.index.max()
    window = pd.Timedelta(minutes=5)

    with PdfPages(pdf_path) as pdf:
        t0 = start_time
        while t0 < end_time:
            t1 = t0 + window

            fig, axs = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
            fig.suptitle(f'{participant_name} - {t0} to {t1}')

            #Plot Nasal Flow
            axs[0].plot(flow_df.loc[t0:t1].index, flow_df.loc[t0:t1]['Value'], color='blue')
            axs[0].set_ylabel('Nasal Flow (L/min)')

            #Overlay events
            for _, row in events_df.iterrows():
                if row['Start'] < t1 and row['End'] > t0:
                    axs[0].axvspan(max(row['Start'], t0), min(row['End'], t1), color='red', alpha=0.3)

            # Plot Thoracic Movement
            axs[1].plot(thor_df.loc[t0:t1].index, thor_df.loc[t0:t1]['Value'], color='green')
            axs[1].set_ylabel('Resp. Amplitude')

            # Plot SpO2
            axs[2].plot(spo2_df.loc[t0:t1].index, spo2_df.loc[t0:t1]['Value'], color='orange')
            axs[2].set_ylabel('SpO2 (%)')
            axs[2].set_xlabel('Time')

            for ax in axs:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                ax.grid(True)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            pdf.savefig(fig)
            plt.close(fig)

            t0 = t1

    print(f'Visualization saved to {pdf_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-name', '--name', required=True, help='Path to participant data folder')
    args = parser.parse_args()

    generate_visualization(args.name)
