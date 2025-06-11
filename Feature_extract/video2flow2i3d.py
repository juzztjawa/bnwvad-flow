import argparse
import sys
import os
import os.path as osp
import glob
from multiprocessing import Pool, Manager, set_start_method # Import set_start_method
import torch
import numpy as np
import cv2
import torchvision.transforms.functional as TF
import torchvision.models.optical_flow as oflow
from torch.autograd import Variable
# Make sure this import path is correct for your environment
from pytorch_i3d import InceptionI3d as I3D
from PIL import Image
from tqdm import tqdm
import math
import traceback # Import traceback module

# --- Configuration ---
CHUNK_SIZE = 16
I3D_FREQUENCY = 16
I3D_INPUT_SIZE = 224
I3D_PREPROCESS_SIZE = (340, 256) # (W, H)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Helper Functions ---
# (resize_flow, oversample_data, forward_batch_i3d remain the same)
# Make sure resize_flow handles potential errors gracefully if needed

def resize_flow(flow_np, target_size=(340, 256)):
    """Resizes a single HxWx2 flow numpy array using PIL."""
    try:
        # Separate channels
        flow_x = Image.fromarray(flow_np[:, :, 0])
        flow_y = Image.fromarray(flow_np[:, :, 1])

        # Resize using LANCZOS
        flow_x_resized = flow_x.resize(target_size, Image.Resampling.LANCZOS)
        flow_y_resized = flow_y.resize(target_size, Image.Resampling.LANCZOS)

        # Convert back to numpy arrays
        flow_x_np = np.array(flow_x_resized).astype(float)
        flow_y_np = np.array(flow_y_resized).astype(float)

        # Stack back into HxWx2
        resized_flow = np.stack([flow_x_np, flow_y_np], axis=-1)
        return resized_flow
    except Exception as e:
        pid = os.getpid()
        print(f"[{pid}] !!! ERROR in resize_flow !!!", file=sys.stderr)
        print(f"[{pid}] Original flow shape: {flow_np.shape if isinstance(flow_np, np.ndarray) else 'Invalid input'}", file=sys.stderr)
        print(f"[{pid}] Target size: {target_size}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr) # Print full traceback to stderr
        sys.stderr.flush()
        # Return None or raise the exception again if you want processing to stop
        return None # Let the caller handle the None value

def oversample_data(data_batch):
    """
    Applies 10-crop oversampling to a batch of chunks.
    Input: (batch_size, chunk_size, H, W, C) numpy array
    Output: List of 10 numpy arrays, each (batch_size, chunk_size, 224, 224, C)
    """
    pid = os.getpid() # Get process ID for logging
    try:
        target_h, target_w = I3D_INPUT_SIZE, I3D_INPUT_SIZE # 224, 224
        # Add check for input shape validity
        if data_batch.ndim != 5:
             raise ValueError(f"[{pid}] Invalid input dimension for oversample_data: {data_batch.ndim}, expected 5")
        if data_batch.shape[2] < target_h or data_batch.shape[3] < target_w:
            raise ValueError(f"[{pid}] Input H/W ({data_batch.shape[2]}x{data_batch.shape[3]}) to oversample_data is smaller than target crop size ({target_h}x{target_w})")

        input_h, input_w = data_batch.shape[2], data_batch.shape[3] # 256, 340

        center_h = (input_h - target_h) // 2
        center_w = (input_w - target_w) // 2
        left_w = 0
        top_h = 0
        right_w = input_w - target_w
        bottom_h = input_h - target_h

        # Ensure indices are valid
        if not (0 <= top_h < input_h and 0 <= left_w < input_w and \
                0 <= bottom_h < input_h and 0 <= right_w < input_w and \
                0 <= center_h < input_h and 0 <= center_w < input_w and \
                top_h+target_h <= input_h and left_w+target_w <= input_w and \
                bottom_h+target_h <= input_h and right_w+target_w <= input_w and \
                center_h+target_h <= input_h and center_w+target_w <= input_w):
            raise ValueError(f"[{pid}] Invalid crop indices calculated in oversample_data. Input: {input_h}x{input_w}, Target: {target_h}x{target_w}")


        # 5 crops
        crop1 = np.array(data_batch[:, :, top_h:top_h+target_h, left_w:left_w+target_w, :])
        crop2 = np.array(data_batch[:, :, top_h:top_h+target_h, right_w:right_w+target_w, :])
        crop3 = np.array(data_batch[:, :, center_h:center_h+target_h, center_w:center_w+target_w, :])
        crop4 = np.array(data_batch[:, :, bottom_h:bottom_h+target_h, left_w:left_w+target_w, :])
        crop5 = np.array(data_batch[:, :, bottom_h:bottom_h+target_h, right_w:right_w+target_w, :])

        # Apply horizontal flip
        data_batch_flip = np.array(data_batch[:, :, :, ::-1, :])

        # 5 flipped crops
        crop6 = np.array(data_batch_flip[:, :, top_h:top_h+target_h, left_w:left_w+target_w, :])
        crop7 = np.array(data_batch_flip[:, :, top_h:top_h+target_h, right_w:right_w+target_w, :])
        crop8 = np.array(data_batch_flip[:, :, center_h:center_h+target_h, center_w:center_w+target_w, :])
        crop9 = np.array(data_batch_flip[:, :, bottom_h:bottom_h+target_h, left_w:left_w+target_w, :])
        crop10 = np.array(data_batch_flip[:, :, bottom_h:bottom_h+target_h, right_w:right_w+target_w, :])

        return [crop1, crop2, crop3, crop4, crop5, crop6, crop7, crop8, crop9, crop10]
    except Exception as e:
        print(f"[{pid}] !!! ERROR in oversample_data !!!", file=sys.stderr)
        print(f"[{pid}] Input batch data shape: {data_batch.shape if isinstance(data_batch, np.ndarray) else 'Invalid input'}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        return None # Signal error to the caller


def forward_batch_i3d(b_data, net):
    """
    Runs I3D inference on a single batch (one crop). Includes error handling.
    """
    pid = os.getpid()
    try:
        # Transpose to: (batch, channels, chunk_size, H, W) expected by I3D
        b_data_t = b_data.transpose([0, 4, 1, 2, 3])
        b_data_t = torch.from_numpy(b_data_t)

        with torch.no_grad():
            b_data_v = Variable(b_data_t.to(DEVICE)).float()
            # print(f"[{pid}] Input tensor to I3D shape: {b_data_v.shape}, Device: {b_data_v.device}", flush=True) # Verbose debug
            b_features = net.extract_features(b_data_v)

        # Remove singleton dimensions and move to CPU
        b_features = b_features.squeeze(4).squeeze(3).squeeze(2)
        b_features_np = b_features.cpu().numpy()
        # print(f"[{pid}] Output features shape: {b_features_np.shape}", flush=True) # Verbose debug
        return b_features_np

    except Exception as e:
        print(f"[{pid}] !!! ERROR in forward_batch_i3d !!!", file=sys.stderr)
        print(f"[{pid}] Input numpy data shape: {b_data.shape if isinstance(b_data, np.ndarray) else 'Invalid input'}", file=sys.stderr)
        # Avoid printing huge tensors to log if it's a CUDA OOM
        if "CUDA out of memory" in str(e):
             print(f"[{pid}] Error details: CUDA out of memory.", file=sys.stderr)
        else:
             traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        return None # Signal error


# --- Main Processing Function ---

def process_video_to_features(vid_item_args):
    """
    Processes a single video: reads frames, computes flow, batches,
    extracts I3D features, and saves them. Includes robust error handling.
    """
    full_path, vid_rel_path, output_dir, i3d_batch_size, i3d_model_path = vid_item_args
    pid = os.getpid() # Get process ID for logging

    try: # Wrap the entire function for unforeseen errors
        # --- Initialization within the worker ---
        print(f"[{pid}] Processing: {vid_rel_path}", flush=True)

        # Load RAFT model
        try:
            raft_model = oflow.raft_small().to(DEVICE).eval()
        except Exception as e:
            print(f"[{pid}] CRITICAL Error loading RAFT model: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            return False # Cannot proceed without RAFT

        # Load I3D model
        try:
            i3d_model = I3D(400, in_channels=2, dropout_keep_prob=0)
            i3d_model.load_state_dict(torch.load(i3d_model_path, map_location=torch.device(DEVICE)))
            i3d_model = i3d_model.to(DEVICE).eval()
        except Exception as e:
            print(f"[{pid}] CRITICAL Error loading I3D model ({i3d_model_path}): {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            return False # Cannot proceed without I3D

        # --- Output file setup ---
        base_name = osp.basename(vid_rel_path).replace(".mp4", "").replace(".avi", "")
        save_file = osp.join(output_dir, f'{base_name}_i3d.npy')

        if osp.exists(save_file):
            print(f"[{pid}] Features exist: {save_file}. Skipping.", flush=True)
            return True

        # --- Video Reading and Flow Calculation ---
        cap = cv2.VideoCapture(full_path)
        if not cap.isOpened():
            print(f"[{pid}] Error opening video: {full_path}", file=sys.stderr, flush=True)
            return False

        ret, prev_frame_bgr = cap.read()
        if not ret:
            print(f"[{pid}] Error reading first frame: {full_path}", file=sys.stderr, flush=True)
            cap.release()
            return False

        prev_frame_rgb = cv2.cvtColor(prev_frame_bgr, cv2.COLOR_BGR2RGB)

        frame_idx = 0
        flow_buffer = []
        chunk_buffer = []
        all_video_features = [[] for _ in range(10)]
        batch_counter = 0 # Keep track of batches processed

        while True:
            ret, curr_frame_bgr = cap.read()
            if not ret:
                print(f"[{pid}] End of video reached at frame {frame_idx}.", flush=True)
                break # End of video

            try: # Wrap frame processing
                curr_frame_rgb = cv2.cvtColor(curr_frame_bgr, cv2.COLOR_BGR2RGB)
                frame1_t = TF.to_tensor(prev_frame_rgb).unsqueeze(0).to(DEVICE)
                frame2_t = TF.to_tensor(curr_frame_rgb).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    flow_tensor = raft_model(frame1_t, frame2_t)[-1]

                flow_np = flow_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                flow_buffer.append(flow_np)

                # --- Chunking ---
                if len(flow_buffer) == CHUNK_SIZE:
                    # print(f"[{pid}] Creating chunk from flow buffer (size {len(flow_buffer)})...", flush=True) # Debug
                    preprocessed_chunk = []
                    valid_chunk = True
                    for idx, flow_frame in enumerate(flow_buffer):
                        resized_flow = resize_flow(flow_frame, target_size=I3D_PREPROCESS_SIZE)
                        if resized_flow is None: # Check if resize failed
                            print(f"[{pid}] Error resizing flow frame {idx} in chunk. Skipping chunk.", file=sys.stderr, flush=True)
                            valid_chunk = False
                            break # Stop processing this chunk
                        preprocessed_chunk.append(resized_flow)

                    if valid_chunk:
                        # Stack into (chunk_size, H, W, C)
                        chunk_array = np.stack(preprocessed_chunk, axis=0)
                        # print(f"[{pid}] Chunk created. Shape: {chunk_array.shape}", flush=True) # Debug
                        chunk_buffer.append(chunk_array)
                    else:
                        # If resize failed, we need to decide how to proceed.
                        # Option: Just discard this chunk's frames from flow_buffer and continue
                        print(f"[{pid}] Discarding invalid chunk.", flush=True)
                        # Clear or slide buffer appropriately based on strategy below

                    # --- Batching and I3D Inference ---
                    if len(chunk_buffer) == i3d_batch_size:
                        batch_counter += 1
                        print(f"[{pid}] Processing batch {batch_counter}. Chunk buffer size: {len(chunk_buffer)}", flush=True)
                        try: # Wrap the whole batch processing block
                            # Stack chunks into a batch: (batch, chunk, H, W, C)
                            batch_data = np.stack(chunk_buffer, axis=0)
                            print(f"[{pid}] Batch {batch_counter} stacked. Shape: {batch_data.shape}", flush=True)

                            batch_data_ten_crop = oversample_data(batch_data)
                            if batch_data_ten_crop is None: # Check if oversample failed
                                print(f"[{pid}] Error during oversampling for batch {batch_counter}. Skipping batch.", file=sys.stderr, flush=True)
                                chunk_buffer = [] # Clear buffer even on error
                                continue # Skip to next video frame read

                            print(f"[{pid}] Batch {batch_counter} oversampling done. Processing 10 crops...", flush=True)

                            # Run I3D inference for each crop
                            batch_failed = False
                            for i in range(10):
                                print(f"[{pid}] Batch {batch_counter}, Crop {i+1}/10: ", end="", flush=True)
                                crop_data = batch_data_ten_crop[i]

                                # Assertion checks (good to keep)
                                assert crop_data.shape[2] == I3D_INPUT_SIZE, f"Crop H mismatch: {crop_data.shape[2]} != {I3D_INPUT_SIZE}"
                                assert crop_data.shape[3] == I3D_INPUT_SIZE, f"Crop W mismatch: {crop_data.shape[3]} != {I3D_INPUT_SIZE}"

                                # Get features: (batch, feature_dim)
                                features = forward_batch_i3d(crop_data, i3d_model)

                                if features is None: # Check if forward pass failed
                                    print(f"\n[{pid}] !!! Forward pass failed for Batch {batch_counter}, Crop {i+1}. Skipping rest of batch. !!!", file=sys.stderr, flush=True)
                                    # Decide strategy: skip entire batch or just try other crops?
                                    # For safety, let's skip the whole batch if one crop fails badly (e.g., OOM likely affects others too)
                                    batch_failed = True
                                    break # Exit the inner loop (over crops)

                                print(f"Feature shape: {features.shape}", flush=True)
                                all_video_features[i].append(features)
                                # print("#"*10, f" Appended crop {i}", "#"*10) # More concise debug

                            if batch_failed:
                                # Need to clean up potential partial appends if needed, but simpler is to just log and clear
                                print(f"[{pid}] Cleaning up after failed batch {batch_counter}.", file=sys.stderr, flush=True)
                                # Ensure features from the failed batch are not kept (they weren't appended if forward_batch_i3d returned None)
                            else:
                                print(f"[{pid}] Batch {batch_counter} processed successfully.", flush=True)

                            # Clear the chunk buffer AFTER processing or error handling
                            chunk_buffer = []

                        except Exception as batch_err:
                            print(f"[{pid}] !!! UNHANDLED ERROR during BATCH {batch_counter} processing !!!", file=sys.stderr)
                            traceback.print_exc(file=sys.stderr)
                            sys.stderr.flush()
                            chunk_buffer = [] # Clear buffer on unexpected error too
                            # Decide if you want to stop the whole video or try to continue
                            # For robustness, let's try to continue with the next frame
                            continue

                    # --- Manage flow_buffer ---
                    # Slide the window (this should happen regardless of whether a chunk was valid or a batch was processed)
                    flow_buffer = flow_buffer[I3D_FREQUENCY:]

            except Exception as frame_err:
                print(f"[{pid}] !!! ERROR processing frame {frame_idx} !!!", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()
                # Decide how to handle frame error: skip frame, stop video?
                # Let's try skipping to the next frame, but update prev_frame cautiously
                # If flow calculation failed, prev_frame might not be reliable for next iter
                # Safer to break if flow calculation fails? For now, let's risk continuing.
                pass # Allows loop to continue, prev_frame is updated below

            # Update previous frame MUST happen outside the try block for flow calculation
            prev_frame_rgb = curr_frame_rgb
            frame_idx += 1
            # Add a small print occasionally to show progress within a long video
            # if frame_idx % 100 == 0:
            #     print(f"[{pid}] Reached frame {frame_idx}", flush=True)


        # --- Process Remaining Chunks ---
        print(f"[{pid}] Processing remaining {len(chunk_buffer)} chunks...", flush=True)
        if chunk_buffer:
            batch_counter += 1
            print(f"[{pid}] Processing final batch {batch_counter} (partial). Chunk buffer size: {len(chunk_buffer)}", flush=True)
            try: # Wrap final batch processing
                batch_data = np.stack(chunk_buffer, axis=0)
                print(f"[{pid}] Final Batch {batch_counter} stacked. Shape: {batch_data.shape}", flush=True)

                batch_data_ten_crop = oversample_data(batch_data)
                if batch_data_ten_crop is None:
                    print(f"[{pid}] Error during oversampling for final batch {batch_counter}. No features saved.", file=sys.stderr, flush=True)
                    # Set features to empty or handle accordingly
                    all_video_features = [[] for _ in range(10)] # Reset features as they are incomplete
                else:
                    print(f"[{pid}] Final Batch {batch_counter} oversampling done. Processing 10 crops...", flush=True)
                    batch_failed = False
                    for i in range(10):
                        print(f"[{pid}] Final Batch {batch_counter}, Crop {i+1}/10: ", end="", flush=True)
                        crop_data = batch_data_ten_crop[i]
                        assert crop_data.shape[2] == I3D_INPUT_SIZE
                        assert crop_data.shape[3] == I3D_INPUT_SIZE

                        features = forward_batch_i3d(crop_data, i3d_model)
                        if features is None:
                            print(f"\n[{pid}] !!! Forward pass failed for Final Batch {batch_counter}, Crop {i+1}. !!!", file=sys.stderr, flush=True)
                            batch_failed = True
                            # Decide if partial features should be saved. Probably not.
                            break # Exit inner loop

                        print(f"Feature shape: {features.shape}", flush=True)
                        all_video_features[i].append(features)

                    if batch_failed:
                        print(f"[{pid}] Final batch processing failed. Features might be incomplete or not saved.", file=sys.stderr, flush=True)
                        all_video_features = [[] for _ in range(10)] # Discard potentially partial features

                chunk_buffer = [] # Clear buffer

            except Exception as final_batch_err:
                 print(f"[{pid}] !!! UNHANDLED ERROR during FINAL BATCH {batch_counter} processing !!!", file=sys.stderr)
                 traceback.print_exc(file=sys.stderr)
                 sys.stderr.flush()
                 all_video_features = [[] for _ in range(10)] # Discard features on error

        cap.release() # Ensure cap is released

        # --- Aggregation and Saving ---
        # Check if *any* features were successfully generated *at all*
        all_appended_features = [
            feature_batch
            for crop_features_list in all_video_features # Iterate through the 10 outer lists
            for feature_batch in crop_features_list      # Iterate through feature arrays (batches) in each inner list
            if feature_batch is not None                 # Ensure it's not None (though should be handled earlier)
        ]

        # Check if the flattened list is empty (meaning no features were successfully appended)
        if not all_appended_features:
            print(f"[{pid}] No features were successfully generated or appended for {vid_rel_path}. Skipping save.", file=sys.stderr, flush=True)
            # Clean up resources before returning False
            del raft_model
            del i3d_model
            if DEVICE == 'cuda': torch.cuda.empty_cache()
            return False # Indicate failure or lack of data

        print(f"[{pid}] Aggregating final features...", flush=True)
        try: # Wrap aggregation and saving
            # Filter out potential None values if error handling allows partial success (safer to fail batch)
            # Concatenate features from all batches for each crop
            final_features_per_crop = []
            min_chunks = float('inf') # Find the minimum number of chunks processed across all crops (in case of errors)
            for crop_idx, crop_features in enumerate(all_video_features):
                if crop_features: # If list is not empty
                    concatenated = np.concatenate(crop_features, axis=0)
                    final_features_per_crop.append(concatenated)
                    min_chunks = min(min_chunks, concatenated.shape[0])
                    # print(f"[{pid}] Crop {crop_idx} concatenated shape: {concatenated.shape}", flush=True) # Debug
                else:
                    # Handle case where a crop consistently failed - cannot stack later
                    print(f"[{pid}] WARNING: Crop {crop_idx} produced no features. Final result might be unusable or saving will fail.", file=sys.stderr, flush=True)
                    # Option 1: Add zeros/placeholder (risky)
                    # Option 2: Make the whole process fail here
                    # Option 3: Exclude this crop (changes output format)
                    # Let's choose to fail if stacking isn't possible
                    final_features_per_crop.append(None) # Mark as None

            # Check if all crops have data and consistent chunk count before stacking
            if any(x is None for x in final_features_per_crop) or len(final_features_per_crop) != 10:
                 print(f"[{pid}] ERROR: Not all 10 crops produced valid features. Cannot stack. Skipping save.", file=sys.stderr, flush=True)
                 return False

            # Ensure all crops have the *same* number of chunks before stacking
            # This might happen if one crop failed on the last batch, but others didn't
            consistent_shape = True
            num_chunks = -1
            if final_features_per_crop:
                num_chunks = final_features_per_crop[0].shape[0]
                for i in range(1, 10):
                    if final_features_per_crop[i].shape[0] != num_chunks:
                        consistent_shape = False
                        print(f"[{pid}] ERROR: Inconsistent number of chunks between crops ({final_features_per_crop[i].shape[0]} vs {num_chunks}). Skipping save.", file=sys.stderr, flush=True)
                        break

            if not consistent_shape:
                return False


            # Stack the 10 crops' features: (10, total_chunks, feature_dim)
            final_features = np.stack(final_features_per_crop, axis=0)

            os.makedirs(osp.dirname(save_file), exist_ok=True)
            np.save(save_file, final_features)
            total_chunks_processed = final_features.shape[1]
            print(f"[{pid}] COMPLETED: {vid_rel_path}. Frames: {frame_idx}. Chunks: {total_chunks_processed}. Saved: {save_file} (Shape: {final_features.shape})", flush=True)
            return True

        except Exception as agg_err:
            print(f"[{pid}] !!! ERROR during feature aggregation or saving for {vid_rel_path} !!!", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            return False # Failed during final step

    except Exception as e:
        # Catch-all for the entire function
        pid = os.getpid() # Ensure pid is defined
        print(f"[{pid}] !!! UNCAUGHT TOP-LEVEL ERROR in process_video_to_features for {vid_rel_path} !!!", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        # Ensure cap is released if open
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        return False # Indicate failure

# --- Argument Parsing ---
# (parse_args function remains the same)
def parse_args():
    parser = argparse.ArgumentParser(description='Extract I3D features from videos using in-memory optical flow')
    parser.add_argument('--src_dir', default="UCF-Crime/", type=str, help='Source directory containing videos.')
    parser.add_argument('--output_dir', default="UCF_Crime_I3D_Features/", type=str, help='Output directory for extracted I3D features (.npy files).')
    parser.add_argument('--ext', type=str, default='mp4', choices=['avi', 'mp4'], help='Video file extension.')
    parser.add_argument('--level', type=int, choices=[1, 2], default=2, help='Directory structure level: 1 (videos in src_dir), 2 (videos in src_dir/class/video).')
    parser.add_argument('--i3d_model_path', default="flow_imagenet.pt", type=str, help='Path to the pre-trained I3D model weights (for flow).')
    parser.add_argument('--i3d_batch_size', type=int, default=20, help='Number of chunks to process in a batch for I3D.')
    parser.add_argument('--num_worker', type=int, default=4, help='Number of worker processes.')
    parser.add_argument("--resume", action='store_true', default=True, help='Resume feature extraction by skipping videos with existing .npy files.')
    return parser.parse_args()


if __name__ == '__main__':
    # Set the start method BEFORE creating Pool
    try:
        set_start_method('spawn', force=True)
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError as e:
        print(f"Warning: Could not set multiprocessing start method to 'spawn': {e}", file=sys.stderr)

    args = parse_args()

    # --- Sanity Checks ---
    if not osp.isdir(args.src_dir):
        print(f"Error: Source directory not found: {args.src_dir}")
        sys.exit(1)
    if not osp.isfile(args.i3d_model_path):
        print(f"Error: I3D model weights not found: {args.i3d_model_path}")
        sys.exit(1)
    if args.num_worker <= 0:
        print(f"Warning: num_worker should be > 0. Setting to 1.")
        args.num_worker = 1
    if CHUNK_SIZE != I3D_FREQUENCY:
        print(f"Warning: CHUNK_SIZE ({CHUNK_SIZE}) and I3D_FREQUENCY ({I3D_FREQUENCY}) are different. Ensure this is intended for sparse sampling or overlap.")


    # Create the output feature folder
    if not osp.isdir(args.output_dir):
        print(f'Creating output directory: {args.output_dir}')
        os.makedirs(args.output_dir)

    # --- Video List Discovery ---
    print(f'Reading videos from: {args.src_dir}, Extension: {args.ext}')
    if args.level == 2:
        fullpath_list = glob.glob(osp.join(args.src_dir, '*', f'*.{args.ext}'))
        # Create corresponding class subdirectories in the output folder
        classes = [d for d in os.listdir(args.src_dir) if osp.isdir(osp.join(args.src_dir, d))]
        for classname in classes:
            class_out_dir = osp.join(args.output_dir, classname)
            if not osp.isdir(class_out_dir):
                print(f'Creating output subdirectory: {class_out_dir}')
                os.makedirs(class_out_dir)
    else: # level 1
        fullpath_list = glob.glob(osp.join(args.src_dir, f'*.{args.ext}'))

    print(f"Found {len(fullpath_list)} videos.")

    # --- Prepare arguments for workers ---
    task_args = []
    skipped_count = 0
    processed_count = 0
    for full_path in fullpath_list:
        if args.level == 2:
            # e.g., UCF-Crime/Arson/Arson001.mp4 -> Arson/Arson001.mp4
            vid_rel_path = osp.join(*(full_path.split(os.sep)[-2:]))
            # e.g., UCF_Crime_I3D_Features/Arson/
            video_output_dir = osp.join(args.output_dir, osp.dirname(vid_rel_path))
        else: # level 1
            # e.g., UCF-Crime/Arson001.mp4 -> Arson001.mp4
            vid_rel_path = osp.basename(full_path)
            # e.g., UCF_Crime_I3D_Features/
            video_output_dir = args.output_dir

        base_name = osp.basename(vid_rel_path).replace(f".{args.ext}", "")
        expected_feature_file = osp.join(video_output_dir, f'{base_name}_i3d.npy')

        # Check for resuming
        if args.resume and osp.exists(expected_feature_file):
            # print(f"Resuming: Skipping {vid_rel_path} as features exist.")
            skipped_count += 1
            continue

        task_args.append((
            full_path,
            vid_rel_path, # Pass relative path for logging/structure
            video_output_dir, # Pass the specific output dir for this video's features
            args.i3d_batch_size,
            args.i3d_model_path
        ))
        processed_count += 1

    print(f"Total videos to process: {processed_count}")
    if args.resume:
        print(f"Skipped {skipped_count} videos due to existing features.")



    # --- Start Multiprocessing ---
    if not task_args:
        print("No videos left to process.")
    else:
        print(f"Starting processing with {args.num_worker} workers...")
        # Use Pool context manager for better resource cleanup
        with Pool(args.num_worker) as pool:
            results = list(tqdm(pool.imap_unordered(process_video_to_features, task_args), total=len(task_args), desc="Processing Videos"))
            # results = pool.map(process_video_to_features, task_args) # Alternative without progress bar

        # Report success/failure
        success_count = sum(1 for r in results if r is True)
        failure_count = len(results) - success_count
        print(f"\nProcessing Finished.")
        print(f"Successfully processed/skipped: {success_count}")
        print(f"Failed during processing: {failure_count}")
        if failure_count > 0:
             print("Check stderr logs for error details from failed processes.")