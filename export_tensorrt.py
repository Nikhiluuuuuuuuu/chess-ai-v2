import torch
import os
from train import DeepVisionElite

# --- CONFIGURATION ---
MODEL_PATH = "chess_model_rl_latest.pth"
ONNX_PATH = "deepvision_elite.onnx"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def export_to_onnx():
    print(f"Loading DeepVisionElite from {MODEL_PATH}...")
    model = DeepVisionElite().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
    else:
        print(f"Warning: {MODEL_PATH} not found. Exporting random weights.")
    
    model.eval()
    
    # Create dummy input: Batch size of 1024 to match our new MCTS batch scale
    # Tensor shape is [Batch, Channels, Height, Width] -> [1024, 32, 8, 8]
    dummy_input = torch.randn(1024, 32, 8, 8, device=DEVICE)
    
    print(f"Exporting model to {ONNX_PATH}...")
    
    # We export with dynamic axes to allow varying batch sizes during inference
    torch.onnx.export(
        model, 
        dummy_input, 
        ONNX_PATH, 
        export_params=True,
        opset_version=14, 
        do_constant_folding=True,
        input_names=['board_state'],
        output_names=['policy', 'wdl', 'aux', 'lookahead', 'dsi', 'mirror', 'dtm'],
        dynamic_axes={
            'board_state': {0: 'batch_size'},
            'policy': {0: 'batch_size'},
            'wdl': {0: 'batch_size'},
            'aux': {0: 'batch_size'},
            'lookahead': {0: 'batch_size'},
            'dsi': {0: 'batch_size'},
            'mirror': {0: 'batch_size'},
            'dtm': {0: 'batch_size'}
        }
    )
    print(f"Export complete: {ONNX_PATH}")
    print("\n" + "="*50)
    print("Next Step: Compile to TensorRT (INT8 Quantization)")
    print("="*50)
    print("Run the following command on your GPU server equipped with TensorRT:")
    print(f"trtexec --onnx={ONNX_PATH} \\")
    print("        --saveEngine=deepvision_elite.trt \\")
    print("        --int8 \\")
    print("        --best \\")
    print("        --optShapes=board_state:1024x32x8x8 \\")
    print("        --maxShapes=board_state:2048x32x8x8 \\")
    print("        --minShapes=board_state:1x32x8x8")
    print("\nThis will generate an aggressively optimized 'deepvision_elite.trt' engine")
    print("capable of millisecond-latency inference utilizing Tensor Cores natively.")

if __name__ == "__main__":
    export_to_onnx()
