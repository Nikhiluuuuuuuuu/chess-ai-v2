import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import chess
import numpy as np
import os
import glob
import time

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "chess_model_rl_latest.pth"
DATA_DIR = "self_play_data"
GAMES_PER_ITERATION = 50       # How many games to play before training
MCTS_SIMULATIONS = 200         # Search depth per move
RL_EPOCHS = 2                  # How many times to sweep the data per train step
BATCH_SIZE = 1024              # Maxed for 24GB L4 GPU
WORKERS = 24                   # Maxed for 28 vCPUs

os.makedirs(DATA_DIR, exist_ok=True)

# ==========================================
# 1. ARCHITECTURE (DEEPVISION ELITE)
# ==========================================
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

class MultiScaleBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1)
        self.conv3x3 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.GELU()
        self.se = SEBlock(in_channels)
    def forward(self, x):
        out = torch.cat([self.conv1x1(x), self.conv3x3(x), self.conv5x5(x)], dim=1)
        return self.relu(self.se(self.bn(out)) + x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Linear(embed_dim * 4, embed_dim))
        self.ln2 = nn.LayerNorm(embed_dim)
        self.spatial_bias = nn.Parameter(torch.zeros(1, num_heads, 64, 64))
    def forward(self, x):
        residual = x
        x = self.ln1(x)
        b, n, d = x.shape
        q, k, v = [x.view(b, n, self.num_heads, self.head_dim).transpose(1, 2) for _ in range(3)]
        attn = torch.softmax(((q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)) + self.spatial_bias, dim=-1)
        x = residual + (attn @ v).transpose(1, 2).contiguous().view(b, n, d)
        return x + self.ffn(self.ln2(x))

class ReasoningLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, embed_dim * 2), nn.GELU(), nn.Linear(embed_dim * 2, embed_dim))
    def forward(self, spatial_features, evaluation_context):
        attn_out, _ = self.cross_attn(spatial_features, evaluation_context, evaluation_context)
        x = self.norm(spatial_features + attn_out)
        return x + self.mlp(x)

class MoEPolicyHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.experts = nn.ModuleList([nn.Sequential(nn.Conv2d(in_channels, 64, 1), nn.BatchNorm2d(64), nn.GELU(), nn.Flatten(), nn.Linear(64*8*8, 4096)) for _ in range(3)])
        self.router = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(in_channels, 32), nn.GELU(), nn.Linear(32, 3), nn.Softmax(dim=-1))
    def forward(self, x):
        weights = self.router(x)
        self.last_router_weights = weights
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        return torch.bmm(weights.unsqueeze(1), expert_outputs).squeeze(1)

class DeepVisionElite(nn.Module):
    def __init__(self, refinement_steps=3):
        super().__init__()
        self.refinement_steps = refinement_steps
        self.input_conv = nn.Sequential(nn.Conv2d(32, 256, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(256), nn.GELU())
        self.blocks = nn.Sequential(*[MultiScaleBlock(256) for _ in range(20)])
        self.attention = nn.ModuleList([TransformerBlock(256, 16) for _ in range(4)])
        self.reasoner = ReasoningLayer(256, num_heads=16)
        self.refinement_mlp = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 256))
        
        self.policy_head = MoEPolicyHead(256)
        self.mirror_head = MoEPolicyHead(256) 
        self.dsi_head = nn.Sequential(nn.Conv2d(256, 32, 1), nn.BatchNorm2d(32), nn.GELU(), nn.Flatten(), nn.Linear(32*8*8, 4096))
        self.value_head = nn.Sequential(nn.Conv2d(256, 8, 1), nn.BatchNorm2d(8), nn.GELU(), nn.Flatten(), nn.Linear(8*8*8, 256), nn.GELU(), nn.Linear(256, 3))
        self.dtm_head = nn.Sequential(nn.Conv2d(256, 8, 1), nn.BatchNorm2d(8), nn.GELU(), nn.Flatten(), nn.Linear(8*8*8, 256), nn.GELU(), nn.Linear(256, 1))
        self.lookahead_head = nn.Sequential(nn.Conv2d(256, 32, 1), nn.BatchNorm2d(32), nn.GELU(), nn.Flatten(), nn.Linear(32*8*8, 512), nn.GELU(), nn.Linear(512, 256))
        self.aux_head = nn.Sequential(nn.Conv2d(256, 4, 1), nn.BatchNorm2d(4), nn.GELU(), nn.Flatten(), nn.Linear(4*8*8, 128), nn.GELU(), nn.Linear(128, 2))

    def forward(self, x):
        x = self.input_conv(x)
        x = self.blocks(x)
        b, c, h, w = x.shape
        x_flat = x.view(b, c, h * w).permute(0, 2, 1)
        for layer in self.attention: x_flat = layer(x_flat)
        for _ in range(self.refinement_steps):
            pooled = torch.mean(x_flat, dim=1, keepdim=True)
            x_flat = x_flat + self.refinement_mlp(self.reasoner(x_flat, pooled))
        x_final = x_flat.permute(0, 2, 1).view(b, c, h, w)
        return self.policy_head(x_final), self.value_head(x_final), self.aux_head(x_final), self.lookahead_head(x_final), self.dsi_head(x_final), self.mirror_head(x_final), self.dtm_head(x_final)

# ==========================================
# 2. UTILS & MCTS ENGINE
# ==========================================
def board_to_tensor_elite(board):
    tensor = np.zeros((32, 8, 8), dtype=np.float32)
    for color in [chess.WHITE, chess.BLACK]:
        for pt in range(1, 7):
            idx = (0 if color == chess.WHITE else 6) + (pt - 1)
            for sq in board.pieces(pt, color): tensor[idx, divmod(sq, 8)] = 1.0
            idx += 12
            for sq in board.pieces(pt, color):
                for a_sq in board.attacks(sq): tensor[idx, divmod(a_sq, 8)] = 1.0
    tensor[24, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
    if board.has_kingside_castling_rights(chess.WHITE): tensor[25, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE): tensor[26, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK): tensor[27, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK): tensor[28, :, :] = 1.0
    tensor[29, :, :] = board.halfmove_clock / 100.0
    if board.ep_square: tensor[30, divmod(board.ep_square, 8)] = 1.0
    tensor[31, :, :] = min(board.fullmove_number, 100) / 100.0
    return torch.from_numpy(tensor).float()

class MCTSNode:
    def __init__(self, board, parent=None, move=None, prior=0):
        self.board = board
        self.parent = parent
        self.move = move
        self.prior = prior
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.is_expanded = False

    def value(self):
        if self.visit_count > 0:
            return self.value_sum / self.visit_count
        return (self.parent.value() - 0.1) if self.parent else 0

    def select_child(self, c_puct=1.4):
        best_score, best_child = -float('inf'), None
        for move, child in self.children.items():
            u_score = c_puct * child.prior * np.sqrt(self.visit_count) / (1 + child.visit_count)
            score = child.value() + u_score
            if score > best_score: best_score, best_child = score, child
        return best_child

class MCTSSearcher:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def search(self, board, max_time=3.0, batch_size=64, alpha=0.0):
        """Used by app.py: Searches for the best move within a time limit."""
        root = MCTSNode(board.copy())
        start_time = time.time()
        self._expand_batch([root], alpha=alpha)
        
        while time.time() - start_time < max_time:
            self._run_batch_iteration(root, batch_size, max_time, start_time, alpha=alpha)
            
        if not root.children: return list(board.legal_moves)[0]
        return max(root.children.items(), key=lambda x: x[1].visit_count)[0]

    def search_for_self_play(self, board, simulations=200, batch_size=64):
        root = MCTSNode(board.copy())
        self._expand_batch([root])
        
        nodes_evaluated = 1
        while nodes_evaluated < simulations:
            leaves_expanded = self._run_batch_iteration(root, batch_size)
            nodes_evaluated += leaves_expanded
        return root

    def _run_batch_iteration(self, root, batch_size, max_time=None, start_time=None, alpha=0.0):
        leaves_to_expand = []
        while len(leaves_to_expand) < batch_size:
            if max_time and (time.time() - start_time >= max_time): break
                
            node = root
            while node.is_expanded and node.children:
                node = node.select_child()
                
            if node.board.is_game_over():
                res = node.board.result()
                val = 1.0 if res == "1-0" else (-1.0 if res == "0-1" else 0)
                self._backpropagate(node, val)
                continue
                
            if node not in leaves_to_expand:
                leaves_to_expand.append(node)
                
        if leaves_to_expand:
            self._expand_batch(leaves_to_expand, alpha=alpha)
        return len(leaves_to_expand)

    def _expand_batch(self, nodes, alpha=0.0):
        tensors = torch.stack([board_to_tensor_elite(n.board) for n in nodes]).to(self.device)
        with torch.no_grad():
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.bfloat16):
                p_out, v_out, _, _, dsi_out, mirror_out, dtm_out = self.model(tensors)
                
            for i, node in enumerate(nodes):
                noise = 0
                if node.parent is None: noise = np.random.dirichlet([0.3] * 4096)
                
                final_logic = p_out[i] + 0.4 * dsi_out[i] - 0.1 * mirror_out[i]
                policy = torch.softmax(final_logic.float(), dim=0).cpu().numpy()
                
                if node.parent is None: policy = 0.75 * policy + 0.25 * noise
                    
                v_logits = v_out[i].float()
                wdl_probs = torch.softmax(v_logits, dim=0)
                value = (wdl_probs[0] * 1.0 + wdl_probs[1] * alpha + wdl_probs[2] * -1.0).item()
                ordered_moves = self._get_ordered_moves(node.board)
                
                for move in ordered_moves:
                    idx = move.from_square * 64 + move.to_square
                    child_board = node.board.copy()
                    child_board.push(move)
                    node.children[move] = MCTSNode(child_board, parent=node, move=move, prior=policy[idx])
                
                node.is_expanded = True
                self._backpropagate(node, value)

    def _backpropagate(self, node, value):
        while node:
            node.visit_count += 1
            node.value_sum += value
            node = node.parent
            value = -value

    def _get_ordered_moves(self, board):
        def score(move):
            s = 0
            if board.is_capture(move): s += 10
            if board.gives_check(move): s += 5
            if move.promotion: s += 8
            return s
        return sorted(list(board.legal_moves), key=score, reverse=True)

# ==========================================
# 3. SELF-PLAY GENERATION
# ==========================================
def encode_policy_target(mcts_root):
    policy = np.zeros(4096, dtype=np.float32)
    total_visits = sum(child.visit_count for child in mcts_root.children.values())
    if total_visits == 0: return policy
    for move, child in mcts_root.children.items():
        idx = move.from_square * 64 + move.to_square
        policy[idx] = child.visit_count / total_visits
    return policy

def play_single_game(mcts_engine):
    board = chess.Board()
    states, policies = [], []
    q_values = []
    move_count = 0
    
    while not board.is_game_over() and move_count < 200:
        mcts_root = mcts_engine.search_for_self_play(board, simulations=MCTS_SIMULATIONS)
        states.append(board_to_tensor_elite(board).numpy())
        policies.append(encode_policy_target(mcts_root))
        q_values.append(mcts_root.value())
        
        moves = list(mcts_root.children.keys())
        visits = [child.visit_count for child in mcts_root.children.values()]
        
        if move_count < 30:
            probs = np.array(visits) / sum(visits)
            best_move = np.random.choice(moves, p=probs)
        else:
            best_move = moves[np.argmax(visits)]
            
        board.push(best_move)
        move_count += 1

    result = board.result()
    winner = 1.0 if result == "1-0" else (-1.0 if result == "0-1" else 0.0)
    values = []
    for i in range(len(states)):
        z = winner * (1.0 if (i % 2 == 0) else -1.0)
        q = q_values[i]
        values.append(0.5 * z + 0.5 * q)
    dtms = [len(states) - i for i in range(len(states))]
    return states, policies, values, dtms

def generate_self_play_data(model):
    model.eval()
    mcts_engine = MCTSSearcher(model, DEVICE) 
    all_states, all_policies, all_values, all_dtms = [], [], [], []
    
    print(f"\n--- GENERATING DATA ({GAMES_PER_ITERATION} GAMES) ---")
    for game_idx in range(GAMES_PER_ITERATION):
        start = time.time()
        s, p, v, d = play_single_game(mcts_engine)
        all_states.extend(s)
        all_policies.extend(p)
        all_values.extend(v)
        all_dtms.extend(d)
        print(f"Game {game_idx+1}/{GAMES_PER_ITERATION} | Moves: {len(s)} | Time: {time.time()-start:.1f}s | Result: {v[0]}")
        
    save_path = os.path.join(DATA_DIR, f"rl_batch_{int(time.time())}.npz")
    np.savez_compressed(
        save_path, 
        boards=np.array(all_states), 
        policies=np.array(all_policies), 
        values=np.array(all_values, dtype=np.float32),
        dtms=np.array(all_dtms, dtype=np.float32)
    )
    print(f"Saved {len(all_states)} positions to {save_path}")

# ==========================================
# 4. REINFORCEMENT LEARNING OPTIMIZER
# ==========================================
class RLDataset(Dataset):
    def __init__(self, data_dir):
        files = glob.glob(os.path.join(data_dir, "*.npz"))
        self.boards, self.policies, self.values, self.dtms = [], [], [], []
        for f in files:
            data = np.load(f)
            self.boards.append(data['boards'])
            self.policies.append(data['policies'])
            self.values.append(data['values'])
            if 'dtms' in data:
                self.dtms.append(data['dtms'])
            else:
                self.dtms.append(np.zeros(len(data['boards']), dtype=np.float32))
            
        self.boards = np.concatenate(self.boards)
        self.policies = np.concatenate(self.policies)
        self.values = np.concatenate(self.values)
        self.dtms = np.concatenate(self.dtms)
        self.num_samples = len(self.boards)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        b = self.boards[idx].copy()
        p = self.policies[idx].copy()
        
        # Dihedral Symmetry Augmentation: 50% chance to flip horizontally
        if np.random.rand() > 0.5:
            b = np.flip(b, axis=2).copy()
            if not hasattr(self, 'flip_map'):
                self.flip_map = np.zeros(4096, dtype=np.int32)
                for f in range(64):
                    for t in range(64):
                        self.flip_map[f * 64 + t] = (f ^ 7) * 64 + (t ^ 7)
            
            p_mirrored = np.zeros_like(p)
            p_mirrored[self.flip_map] = p
            p = p_mirrored

        return (
            torch.from_numpy(b), 
            torch.from_numpy(p), 
            torch.tensor([self.values[idx]], dtype=torch.float32),
            torch.tensor([self.dtms[idx]], dtype=torch.float32)
        )

def train_rl_step(model):
    dataset = RLDataset(DATA_DIR)
    print(f"\n--- TRAINING ON {dataset.num_samples} POSITIONS ---")
    
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, pin_memory=True)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4) 
    scaler = torch.amp.GradScaler('cuda')
    v_crit = nn.MSELoss()
    
    def policy_loss(logits, targets):
        return torch.sum(-targets * torch.nn.functional.log_softmax(logits, dim=1), dim=1).mean()
    
    model.train()
    for epoch in range(RL_EPOCHS): 
        start_time = time.time()
        for i, (s, p_target, v_target, dtm_target) in enumerate(loader):
            s, p_target, v_target, dtm_target = s.to(DEVICE), p_target.to(DEVICE), v_target.to(DEVICE), dtm_target.to(DEVICE)
            s = s.to(memory_format=torch.channels_last)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                p_out, v_out, _, _, _, _, dtm_out = model(s) 
                
                weights_policy = model.policy_head.last_router_weights
                weights_mirror = model.mirror_head.last_router_weights
                
                N_exp = weights_policy.size(1)
                P_policy = weights_policy.mean(dim=0)
                loss_balance_policy = N_exp * torch.sum(P_policy * P_policy)
                
                P_mirror = weights_mirror.mean(dim=0)
                loss_balance_mirror = N_exp * torch.sum(P_mirror * P_mirror)
                
                loss_balance = loss_balance_policy + loss_balance_mirror
                
                v_target_wdl = torch.zeros(v_target.size(0), dtype=torch.long, device=DEVICE)
                v_target_wdl[v_target.squeeze() == 1.0] = 0
                v_target_wdl[v_target.squeeze() == 0.0] = 1
                v_target_wdl[v_target.squeeze() == -1.0] = 2
                v_crit_wdl = nn.CrossEntropyLoss()
                dtm_crit = nn.MSELoss()
                
                loss_v = v_crit_wdl(v_out, v_target_wdl)
                loss_dtm = dtm_crit(dtm_out, dtm_target)
                
                loss = policy_loss(p_out, p_target) + loss_v + 0.1 * loss_dtm + 0.01 * loss_balance
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if i % 50 == 0:
                print(f"Epoch {epoch+1}/{RL_EPOCHS} | Batch {i} | Loss: {loss.item():.4f}")
                
    torch.save(model.state_dict(), MODEL_PATH)
    print("Brain Upgraded. Weights Saved.")

# ==========================================
# 5. THE INFINITE LOOP
# ==========================================
if __name__ == "__main__":
    print(f"Initializing RL Engine on {DEVICE}...")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')

    model = DeepVisionElite().to(DEVICE, memory_format=torch.channels_last)
    
    if os.path.exists(MODEL_PATH):
        print(f"Resuming from {MODEL_PATH}...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
    else:
        print("No prior model found. Starting from scratch (Random Weights).")

    if torch.cuda.is_available():
        try:
            model = torch.compile(model)
            print("PyTorch 2.0 Compilation Successful.")
        except Exception as e:
            print(f"Compilation skipped: {e}")

    iteration = 1
    while True:
        print(f"\n\n{'='*40}\n STARTING RL ITERATION {iteration}\n{'='*40}")
        generate_self_play_data(model)
        train_rl_step(model)
        iteration += 1
        