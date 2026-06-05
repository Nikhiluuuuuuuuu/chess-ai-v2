import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import chess
import numpy as np
import os
import re
import time
import glob

# --- ADVANCED MODEL COMPONENTS ---
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += residual
        return self.relu(out)

class FlashAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        residual = x
        x = self.ln1(x)
        q = x.view(x.size(0), x.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        k = x.view(x.size(0), x.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        v = x.view(x.size(0), x.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(x.size(0), x.size(1), -1)
        x = residual + attn_out
        x = x + self.ffn(self.ln2(x))
        return x

class DeepVision20M(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv2d(140, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(256) for _ in range(16)])
        self.attention = nn.ModuleList([FlashAttentionBlock(256, 8) for _ in range(4)])
        self.policy_head = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 4672)
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(256, 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(4 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh()
        )
        self.complexity_head = nn.Sequential(
            nn.Conv2d(256, 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(4 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )
        self.material_head = nn.Sequential(
            nn.Conv2d(256, 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(4 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.input_conv(x)
        x = self.res_blocks(x)
        b, c, h, w = x.shape
        x_flat = x.view(b, c, h * w).permute(0, 2, 1)
        for layer in self.attention:
            x_flat = layer(x_flat)
        x = x_flat.permute(0, 2, 1).view(b, c, h, w)
        return self.policy_head(x), self.value_head(x), self.complexity_head(x), self.material_head(x)

# --- UTILITIES ---
def board_to_tensor(board):
    tensor = np.zeros((140, 8, 8), dtype=np.float32)
    for color in [chess.WHITE, chess.BLACK]:
        for piece_type in range(1, 7):
            p_idx = (0 if color == chess.WHITE else 6) + (piece_type - 1)
            for square in board.pieces(piece_type, color):
                r, c = divmod(square, 8)
                tensor[p_idx, r, c] = 1.0
                attacks = board.attacks(square)
                for a_sq in attacks:
                    ar, ac = divmod(a_sq, 8)
                    tensor[12 + p_idx, ar, ac] = 1.0
    # King Tropism (24-25)
    for color in [chess.WHITE, chess.BLACK]:
        ksq = board.king(color)
        if ksq is not None:
            kr, kc = divmod(ksq, 8)
            p_idx = 24 if color == chess.WHITE else 25
            for sq in chess.SQUARES:
                piece = board.piece_at(sq)
                if piece and piece.color != color:
                    r, c = divmod(sq, 8)
                    tensor[p_idx, r, c] = max(abs(r-kr), abs(c-kc)) / 7.0
    # Tactical Awareness (132-135)
    for sq in chess.SQUARES:
        r, c = divmod(sq, 8)
        if board.is_pinned(chess.WHITE, sq): tensor[132, r, c] = 1.0
        if board.is_pinned(chess.BLACK, sq): tensor[133, r, c] = 1.0
        if board.is_attacked_by(chess.BLACK, sq): tensor[134, r, c] = 1.0
        if board.is_attacked_by(chess.WHITE, sq): tensor[135, r, c] = 1.0
    tensor[128, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
    return torch.from_numpy(tensor)

def encode_move(move):
    return (move.from_square * 64 + move.to_square) % 4672

def clean_san_sequence(san_str):
    san_str = re.sub(r'\{.*?\}', '', san_str)
    san_str = re.sub(r'\$\d+', '', san_str)
    san_str = re.sub(r'\d+\.{1,3}', '', san_str)
    tokens = san_str.split()
    res = ['1-0', '0-1', '1/2-1/2', '*', '0–1', '1–0', '½–½']
    return [t for t in tokens if t not in res and len(t) > 0]

class ChessCSVDataset(Dataset):
    def __init__(self, csv_path, num_samples=10000000, min_elo=2000):
        self.csv_path, self.num_samples, self.min_elo = csv_path, num_samples, min_elo
        self.offsets = []
        self._index_data()
    def _index_data(self):
        if not os.path.exists(self.csv_path): return
        cache_path = self.csv_path + ".offsets.npy"
        if os.path.exists(cache_path):
            print(f"Loading offsets from cache: {cache_path}")
            self.offsets = np.load(cache_path).tolist()
            if len(self.offsets) > self.num_samples:
                self.offsets = self.offsets[:self.num_samples]
            print(f"Loaded {len(self.offsets)} games from cache.")
            return

        print(f"Indexing {self.csv_path} (this only happens once)...")
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            f.readline()
            pos = f.tell()
            line = f.readline()
            while line and len(self.offsets) < self.num_samples:
                self.offsets.append(pos)
                pos = f.tell()
                line = f.readline()
                if len(self.offsets) % 1000000 == 0: print(f"Indexed {len(self.offsets)} games...")
        
        try:
            np.save(cache_path, np.array(self.offsets, dtype=np.int64))
            print(f"Saved offsets to cache: {cache_path}")
        except Exception as e:
            print(f"Warning: Could not save offsets cache: {e}")
    def __len__(self): return len(self.offsets)
    def __getitem__(self, idx):
        try:
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                f.seek(self.offsets[idx]); line = f.readline()
            parts = line.strip().split(',')
            w_elo = int(parts[6]) if parts[6].isdigit() else 0
            b_elo = int(parts[7]) if parts[7].isdigit() else 0
            if w_elo < self.min_elo or b_elo < self.min_elo: return self.__getitem__(np.random.randint(0, len(self.offsets)))
            weight = min(2.0, max(1.0, ((w_elo + b_elo)/2.0 - 2000) / 1000.0 + 1.0))
            val = 1.0 if parts[3] == '1-0' else (-1.0 if parts[3] == '0-1' else 0.0)
            board = chess.Board(); moves = clean_san_sequence(parts[14])
            stop = np.random.randint(0, len(moves)); moves_left = len(moves) - stop
            target_move = None
            for i, m in enumerate(moves):
                if i >= stop:
                    try: target_move = board.parse_san(m)
                    except: target_move = None
                    break
                try: board.push_san(m)
                except: break
            mat_score = 0
            values = {1:1, 2:3, 3:3, 4:5, 5:9}
            for pt, v in values.items():
                mat_score += len(board.pieces(pt, chess.WHITE)) * v
                mat_score -= len(board.pieces(pt, chess.BLACK)) * v
            policy = np.zeros(4672, dtype=np.float32)
            if target_move: policy[encode_move(target_move)] = 1.0
            v_target = torch.tensor([val if board.turn == chess.WHITE else -val], dtype=torch.float32)
            return (board_to_tensor(board), torch.from_numpy(policy), v_target, 
                    torch.tensor([weight], dtype=torch.float32),
                    torch.tensor([moves_left / 100.0], dtype=torch.float32),
                    torch.tensor([mat_score / 15.0], dtype=torch.float32))
        except: return self.__getitem__(np.random.randint(0, len(self.offsets)))

class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema = DeepVision20M().to(next(model.parameters()).device)
        self.ema.load_state_dict(model.state_dict()); self.decay = decay
    def update(self, model):
        with torch.no_grad():
            for ep, p in zip(self.ema.parameters(), model.parameters()):
                ep.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepVision20M().to(device, memory_format=torch.channels_last); ema = ModelEMA(model)
    if os.path.exists("chess_model_ema_latest.pth"):
        model.load_state_dict(torch.load("chess_model_ema_latest.pth", map_location=device), strict=False)
    
    # torch.compile fix for Windows/Missing Compiler
    if hasattr(torch, "compile"):
        import subprocess
        has_compiler = False
        try:
            subprocess.run(["cl"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            has_compiler = True
        except FileNotFoundError:
            pass
        
        if has_compiler:
            print("Enabling torch.compile...")
            model = torch.compile(model, mode="reduce-overhead")
        else:
            print("Skipping torch.compile (cl.exe not found).")
    # Robust path discovery
    possible_paths = [
        "/kaggle/input/datasets/arevel/chess-games/chess_games.csv",
        "/kaggle/input/chess-games/chess_games.csv", 
        "chess_games.csv"
    ]
    csv_path = next((p for p in possible_paths if os.path.exists(p)), None)
    
    if csv_path is None:
        import glob
        matches = glob.glob("/kaggle/input/**/*.csv", recursive=True)
        if matches:
            csv_path = matches[0]
            print(f"Auto-detected dataset at: {csv_path}")
        else:
            # Fallback to local if on a different machine
            print("Warning: Dataset not found. Using dummy path for initialization.")
            csv_path = "chess_games.csv"

    dataset = ChessCSVDataset(csv_path, num_samples=int(os.environ.get("CHESS_SAMPLES", 1000000)))
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=(device.type == 'cuda'), persistent_workers=True)
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=8e-4, total_steps=len(loader)*7)
    scaler = torch.amp.GradScaler(device.type) if device.type == 'cuda' else None
    p_crit = nn.CrossEntropyLoss(label_smoothing=0.1, reduction='none')
    v_crit = nn.MSELoss(reduction='none'); aux_crit = nn.MSELoss()
    print(f"Starting Elite Strategic Training...")
    for epoch in range(7):
        model.train()
        for i, (s, p, v, w, ml, ms) in enumerate(loader):
            s, p, v, w, ml, ms = [x.to(device, non_blocking=True) for x in [s, p, v, w, ml, ms]]
            s = s.to(memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=(device.type == 'cuda')):
                p_out, v_out, ml_out, ms_out = model(s)
                loss = (p_crit(p_out, p) * w.squeeze()).mean() + (v_crit(v_out, v) * w).mean() + 0.1 * aux_crit(ml_out, ml) + 0.1 * aux_crit(ms_out, ms)
            
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            scheduler.step(); ema.update(model)
            if i % 100 == 0:
                print(f"Ep {epoch+1} | B {i} | Loss: {loss.item():.4f} | MatErr: {aux_crit(ms_out, ms).item():.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
            if i % 1000 == 0 and i > 0:
                torch.save(model.state_dict(), "chess_model_latest_v20m.pth")
                torch.save(ema.ema.state_dict(), "chess_model_ema_latest.pth")

if __name__ == "__main__":
    train()
