import os
import time
import requests
import torch
import numpy as np
from train import DeepVisionElite, play_single_game, DATA_DIR, MCTSSearcher

# --- CONFIGURATION ---
WORKER_ID = f"worker_{int(time.time())}"
SERVER_URL = "http://localhost:8000"  # Replace with actual central server URL
GAMES_PER_BATCH = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def download_latest_weights():
    # In a real distributed system, we would fetch weights from the central server bucket
    # For now, we simulate this by loading the local file
    model_path = "chess_model_rl_latest.pth"
    print(f"[{WORKER_ID}] Downloading latest weights from {SERVER_URL}/weights ...")
    time.sleep(1) # Simulate network latency
    
    model = DeepVisionElite().to(DEVICE)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE), strict=False)
        print(f"[{WORKER_ID}] Weights loaded successfully.")
    else:
        print(f"[{WORKER_ID}] No weights found locally, starting with random weights.")
    
    model.eval()
    if torch.cuda.is_available():
        model = torch.compile(model)
    return model

def upload_data(filepath):
    # Simulate an HTTP POST to the central server
    print(f"[{WORKER_ID}] Uploading {filepath} to {SERVER_URL}/upload ...")
    time.sleep(0.5) # Simulate network transfer
    # requests.post(f"{SERVER_URL}/upload", files={'file': open(filepath, 'rb')})
    print(f"[{WORKER_ID}] Upload complete!")

def run_worker_loop():
    print(f"=== Starting Distributed Worker {WORKER_ID} on {DEVICE} ===")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    while True:
        model = download_latest_weights()
        mcts_engine = MCTSSearcher(model, DEVICE)
        
        all_states, all_policies, all_values, all_dtms = [], [], [], []
        
        print(f"[{WORKER_ID}] Starting generation of {GAMES_PER_BATCH} games...")
        for game_idx in range(GAMES_PER_BATCH):
            s, p, v, d = play_single_game(mcts_engine)
            all_states.extend(s)
            all_policies.extend(p)
            all_values.extend(v)
            all_dtms.extend(d)
            print(f"[{WORKER_ID}] Game {game_idx+1}/{GAMES_PER_BATCH} complete. Result: {v[0]}")
            
        # Save batch
        save_path = os.path.join(DATA_DIR, f"worker_{WORKER_ID}_batch_{int(time.time())}.npz")
        np.savez_compressed(
            save_path, 
            boards=np.array(all_states), 
            policies=np.array(all_policies), 
            values=np.array(all_values, dtype=np.float32),
            dtms=np.array(all_dtms, dtype=np.float32)
        )
        
        upload_data(save_path)
        print(f"[{WORKER_ID}] Batch completed. Ready for next cycle.")

if __name__ == "__main__":
    try:
        run_worker_loop()
    except KeyboardInterrupt:
        print(f"[{WORKER_ID}] Worker shutting down...")
