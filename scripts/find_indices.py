import numpy as np
import datetime

data = np.load('data/BTCUSDT_parquet_neurobars.npz')
timestamps = data['timestamps']

start_2024 = datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc).timestamp()
end_2025 = datetime.datetime(2025, 12, 31, 23, 59, 59, tzinfo=datetime.timezone.utc).timestamp()

start_idx = np.searchsorted(timestamps, start_2024)
end_idx = np.searchsorted(timestamps, end_2025, side='right') - 1

print(f"Start 2024 idx: {start_idx} (Timestamp: {timestamps[start_idx]})")
if end_idx < len(timestamps):
    print(f"End 2025 idx: {end_idx} (Timestamp: {timestamps[end_idx]})")
else:
    print(f"End 2025 idx: {len(timestamps)-1} (Timestamp: {timestamps[-1]})")

print(f"Total bars in 2024-2025: {end_idx - start_idx + 1}")
