import os
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial

input_root = 'UCF_ten_4'
output_root = 'UCF_Crime_I3D_Features'

# Set desired number of worker processes (e.g., 2 or 4)
NUM_WORKERS = 2


# Set number of worker processes
NUM_WORKERS = 4

def process_npy_file(file_path, input_root, output_root):
    rel_path = os.path.relpath(file_path, input_root)  # e.g. Abuse/Abuse001_x264_i3d.npy
    rel_dir = os.path.dirname(rel_path)                # e.g. Abuse
    file = os.path.basename(file_path)                 # e.g. Abuse001_x264_i3d.npy

    try:
        arr = np.load(file_path)
    except Exception as e:
        print(f"⚠️ Error loading {file_path}: {e}")
        return

    if arr.ndim != 3 or arr.shape[0] != 10:
        print(f"⚠️ Skipping {file_path}: shape is {arr.shape}, expected (10, x, 1024)")
        return

    # Prepare output directory and filename
    output_dir = os.path.join(output_root, rel_dir)
    os.makedirs(output_dir, exist_ok=True)
    base_name = file.replace('_i3d', '').replace('.npy', '')

    # Save each of the 10 (x,1024) slices
    for i in range(10):
        slice_arr = arr[i]
        out_filename = f"{base_name}.npy" if i == 0 else f"{base_name}__{i}.npy"
        out_path = os.path.join(output_dir, out_filename)
        np.save(out_path, slice_arr)

    print(f"✅ Processed: {file_path}")

def get_all_npy_files(root):
    npy_files = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.endswith('.npy'):
                npy_files.append(os.path.join(dirpath, f))
    return npy_files

if __name__ == '__main__':
    npy_files = get_all_npy_files(input_root)
    print(f"📦 Found {len(npy_files)} files. Using {NUM_WORKERS} worker(s).")

    with Pool(processes=NUM_WORKERS) as pool:
        pool.map(partial(process_npy_file, input_root=input_root, output_root=output_root), npy_files)
