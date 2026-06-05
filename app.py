from flask import Flask, request, jsonify, render_template
import torch
import chess
import os
import time
from pyngrok import ngrok
from dotenv import load_dotenv
from flask_socketio import SocketIO, emit

# IMPORT THE ENGINE
from kaggle_train_script import DeepVisionElite, board_to_tensor_elite, MCTSSearcher

load_dotenv()
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

ACTIVE_SESSIONS = {}  
GLOBAL_MOVE_HISTORY = [] 

# --- INITIALIZE THE RL ENGINE ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "chess_model_rl_latest.pth"

print(f"Initializing Engine on {DEVICE}...")
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')

model = DeepVisionElite().to(DEVICE, memory_format=torch.channels_last)

if os.path.exists(MODEL_PATH):
    print(f"Loading RL Model from {MODEL_PATH}...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
else:
    print(f"WARNING: {MODEL_PATH} not found. AI will play with random weights!")

model.eval()
if torch.cuda.is_available():
    try:
        model = torch.compile(model)
        print("Model compiled for blazing fast inference.")
    except Exception as e:
        print(f"Compilation skipped: {e}")

mcts_engine = MCTSSearcher(model, DEVICE)

# --- FLASK ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/make_move', methods=['POST'])
def make_move():
    data = request.json
    fen = data.get('fen')
    human_only = data.get('human_only', False)
    
    try:
        board = chess.Board(fen)
        sid = request.args.get('sid', 'anonymous')
        now = time.time()
        
        # State tracking for UI
        if sid not in ACTIVE_SESSIONS:
            ACTIVE_SESSIONS[sid] = {'start_time': now, 'moves': []}
        ACTIVE_SESSIONS[sid]['last_active'] = now
        ACTIVE_SESSIONS[sid]['last_fen'] = board.fen()
        next_turn = 'White' if board.turn == chess.WHITE else 'Black'
        ACTIVE_SESSIONS[sid]['last_turn'] = next_turn + (' (Human)' if human_only or board.turn == chess.WHITE else ' (AI)')
        ACTIVE_SESSIONS[sid]['last_check'] = board.is_check()
        ACTIVE_SESSIONS[sid]['last_over'] = board.is_game_over()

        socketio.emit('live_update', {
            'sid': sid, 'move': 'Move Made', 'fen': board.fen(),
            'eval': ACTIVE_SESSIONS[sid].get('last_eval', 0.0),
            'turn': ACTIVE_SESSIONS[sid]['last_turn'], 'in_check': ACTIVE_SESSIONS[sid]['last_check'],
            'is_over': ACTIVE_SESSIONS[sid]['last_over'], 'total_moves': len(ACTIVE_SESSIONS[sid]['moves']),
            'active_users': len(ACTIVE_SESSIONS), 'global_total': len(GLOBAL_MOVE_HISTORY)
        }, namespace='/admin')

        if human_only: return jsonify({'status': 'broadcasted'})

        # AI MCTS Search (Evaluates 64 boards at a time for 3 seconds)
        ai_move = mcts_engine.search(board, max_time=3.0, batch_size=64)
        board.push(ai_move)
        
        # Final Evaluation for UI metrics
        state_tensor = board_to_tensor_elite(board).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.bfloat16):
                outputs = model(state_tensor)
                v_out, aux_out = outputs[1], outputs[2]
        
        move_info = {
            'sid': sid, 'move': ai_move.uci(), 'evaluation': float(v_out.item()),
            'moves_left': float(aux_out[0, 0].item()), 'material': float(aux_out[0, 1].item()), 'timestamp': now
        }
        
        # Update Session
        ACTIVE_SESSIONS[sid]['moves'].append(move_info)
        ACTIVE_SESSIONS[sid]['last_fen'] = board.fen()
        ACTIVE_SESSIONS[sid]['last_eval'] = move_info['evaluation']
        ACTIVE_SESSIONS[sid]['last_turn'] = 'White (Human)'
        ACTIVE_SESSIONS[sid]['last_check'] = board.is_check()
        ACTIVE_SESSIONS[sid]['last_over'] = board.is_game_over()
        
        GLOBAL_MOVE_HISTORY.append(move_info)
        if len(GLOBAL_MOVE_HISTORY) > 100: GLOBAL_MOVE_HISTORY.pop(0)

        expired = [s for s, data in ACTIVE_SESSIONS.items() if now - data.get('last_active', 0) > 1800]
        for s in expired: del ACTIVE_SESSIONS[s]
        
        socketio.emit('live_update', {
            'sid': sid, 'move': ai_move.uci(), 'fen': board.fen(), 'eval': move_info['evaluation'],
            'turn': 'White (Human)', 'in_check': ACTIVE_SESSIONS[sid]['last_check'],
            'is_over': ACTIVE_SESSIONS[sid]['last_over'], 'total_moves': len(ACTIVE_SESSIONS[sid]['moves']),
            'active_users': len(ACTIVE_SESSIONS), 'global_total': len(GLOBAL_MOVE_HISTORY)
        }, namespace='/admin')

        return jsonify(move_info)
            
    except Exception as e:
        import random
        print(f"Engine failed, playing random move. Error: {e}")
        fallback = random.choice(list(board.legal_moves))
        return jsonify({'move': fallback.uci(), 'evaluation': 0.0, 'moves_left': 0, 'material': 0})

@app.route('/admin')
def admin(): return render_template('admin.html')

@socketio.on('connect', namespace='/admin')
def admin_connect():
    emit('init_state', {'sessions': ACTIVE_SESSIONS, 'active_users': len(ACTIVE_SESSIONS)})

if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    NGROK_AUTH_TOKEN = os.environ.get("NGROK_AUTH_TOKEN")
    if NGROK_AUTH_TOKEN:
        ngrok.set_auth_token(NGROK_AUTH_TOKEN)
        try: print(f" * Public URL: {ngrok.connect(5000).public_url}")
        except Exception as e: print(f" * Could not start ngrok: {e}")
    socketio.run(app, debug=True, port=5000, use_reloader=False)