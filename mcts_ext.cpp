#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <string>
#include <cmath>
#include <unordered_map>
#include <memory>
#include <algorithm>
#include "chess.hpp"

namespace py = pybind11;

struct Node {
    chess::Board board;
    Node* parent;
    chess::Move move;
    float prior;
    float value_sum;
    int visit_count;
    int virtual_loss;
    bool is_expanded;
    std::vector<std::pair<chess::Move, Node*>> children;

    Node(chess::Board b, Node* p = nullptr, chess::Move m = chess::Move::NULL_MOVE, float pr = 0.0f)
        : board(b), parent(p), move(m), prior(pr), value_sum(0.0f), visit_count(0), virtual_loss(0), is_expanded(false) {}

    // No destructor needed for children anymore because Searcher's node_pool handles memory allocation and cleanup

    float value() const {
        int total_visits = visit_count + virtual_loss;
        if (total_visits > 0) return (value_sum - virtual_loss) / total_visits;
        if (parent) return parent->value() - 0.1f; // FPU
        return 0.0f;
    }

    Node* select_child(float c_puct = 1.4f) {
        float best_score = -1e9f;
        Node* best_child = nullptr;

        // Progressive Widening logic
        int k = 3 + visit_count / 2;
        int limit = std::min((int)children.size(), k);

        for (int i = 0; i < limit; ++i) {
            Node* child = children[i].second;
            int total_visits = child->visit_count + child->virtual_loss;
            float u_score = c_puct * child->prior * std::sqrt((float)(visit_count + virtual_loss)) / (1.0f + total_visits);
            float score = child->value() + u_score;
            if (score > best_score) {
                best_score = score;
                best_child = child;
            }
        }
        
        // Apply virtual loss immediately on descent
        if (best_child) {
            best_child->virtual_loss += 1;
        }
        
        return best_child;
    }
};

class Searcher {
private:
    Node* root;
    std::vector<std::unique_ptr<Node>> node_pool;
    std::unordered_map<uint64_t, Node*> transposition_table;

public:
    Searcher() : root(nullptr) {}
    ~Searcher() {} // node_pool automatically cleans up DAG memory safely

    Node* allocate_node(chess::Board b, Node* p = nullptr, chess::Move m = chess::Move::NULL_MOVE, float pr = 0.0f) {
        uint64_t hash = b.hash();
        if (transposition_table.find(hash) != transposition_table.end()) {
            return transposition_table[hash]; // Return DAG pointer from TT
        }
        auto node = std::make_unique<Node>(b, p, m, pr);
        Node* ptr = node.get();
        node_pool.push_back(std::move(node));
        transposition_table[hash] = ptr;
        return ptr;
    }

    void set_root(const std::string& fen) {
        node_pool.clear();
        transposition_table.clear();
        chess::Board b(fen);
        root = allocate_node(b);
    }

    std::vector<std::string> get_leaf_nodes(int batch_size) {
        std::vector<std::string> fens;
        for (int i=0; i<batch_size; ++i) {
            fens.push_back(root->board.getFen());
        }
        return fens;
    }

    void expand_and_backprop(const std::vector<std::vector<float>>& policies, const std::vector<float>& values) {
        root->is_expanded = true;
        root->visit_count += 1;
        root->value_sum += values[0];
    }

    std::string get_best_move() {
        if (!root || root->children.empty()) {
            chess::Movelist moves;
            chess::movegen::legalmoves(moves, root->board);
            if (moves.empty()) return "0000";
            return chess::uci::moveToUci(moves[0]);
        }

        Node* best = nullptr;
        chess::Move best_move = chess::Move::NULL_MOVE;
        int max_visits = -1;
        for (auto& pair : root->children) {
            if (pair.second->visit_count > max_visits) {
                max_visits = pair.second->visit_count;
                best = pair.second;
                best_move = pair.first;
            }
        }
        return chess::uci::moveToUci(best_move);
    }
};

// C++ Tensor Encoding Logic
py::array_t<float> board_to_tensor_elite(const std::string& fen) {
    chess::Board board(fen);
    auto result = py::array_t<float>({32, 8, 8});
    auto buf = result.request();
    float* ptr = (float*)buf.ptr;
    std::fill(ptr, ptr + 32 * 8 * 8, 0.0f);

    auto set_sq = [&](int channel, int sq, float val) {
        int r = sq / 8;
        int f = sq % 8;
        ptr[channel * 64 + r * 8 + f] = val;
    };
    
    // Pieces and attacks
    for (int c = 0; c < 2; ++c) {
        chess::Color color = (c == 0) ? chess::Color::WHITE : chess::Color::BLACK;
        for (int pt = 0; pt < 6; ++pt) {
            int idx = (color == chess::Color::WHITE ? 0 : 6) + pt;
            chess::PieceType pieceType = chess::PieceType(static_cast<chess::PieceType::underlying>(pt));
            
            auto bb = board.pieces(pieceType, color);
            while (bb) {
                int sq = bb.pop();
                set_sq(idx, sq, 1.0f);
            }
            
            // Attacks
            int atk_idx = idx + 12;
            auto piece_bb = board.pieces(pieceType, color);
            while (piece_bb) {
                int sq = piece_bb.pop();
                chess::Square square = static_cast<chess::Square>(sq);
                chess::Bitboard attacks;
                if (pieceType == chess::PieceType::PAWN) attacks = chess::attacks::pawn(color, square);
                else if (pieceType == chess::PieceType::KNIGHT) attacks = chess::attacks::knight(square);
                else if (pieceType == chess::PieceType::BISHOP) attacks = chess::attacks::bishop(square, board.occ());
                else if (pieceType == chess::PieceType::ROOK) attacks = chess::attacks::rook(square, board.occ());
                else if (pieceType == chess::PieceType::QUEEN) attacks = chess::attacks::queen(square, board.occ());
                else if (pieceType == chess::PieceType::KING) attacks = chess::attacks::king(square);
                
                while (attacks) {
                    int a_sq = attacks.pop();
                    set_sq(atk_idx, a_sq, 1.0f);
                }
            }
        }
    }
    
    // Channel 24: Turn
    if (board.sideToMove() == chess::Color::WHITE) {
        for (int i=0; i<64; ++i) ptr[24 * 64 + i] = 1.0f;
    }
    
    // Castling
    if (board.castlingRights().has(chess::Color::WHITE, chess::Board::CastlingRights::Side::KING_SIDE)) {
        for (int i=0; i<64; ++i) ptr[25 * 64 + i] = 1.0f;
    }
    if (board.castlingRights().has(chess::Color::WHITE, chess::Board::CastlingRights::Side::QUEEN_SIDE)) {
        for (int i=0; i<64; ++i) ptr[26 * 64 + i] = 1.0f;
    }
    if (board.castlingRights().has(chess::Color::BLACK, chess::Board::CastlingRights::Side::KING_SIDE)) {
        for (int i=0; i<64; ++i) ptr[27 * 64 + i] = 1.0f;
    }
    if (board.castlingRights().has(chess::Color::BLACK, chess::Board::CastlingRights::Side::QUEEN_SIDE)) {
        for (int i=0; i<64; ++i) ptr[28 * 64 + i] = 1.0f;
    }
    
    // Halfmove
    for (int i=0; i<64; ++i) ptr[29 * 64 + i] = board.halfMoveClock() / 100.0f;
    
    // En Passant
    if (board.enpassantSq() != chess::Square::NO_SQ) {
        set_sq(30, board.enpassantSq().index(), 1.0f);
    }
    
    // Fullmove
    float fm = std::min((int)board.fullMoveNumber(), 100) / 100.0f;
    for (int i=0; i<64; ++i) ptr[31 * 64 + i] = fm;
    
    return result;
}

PYBIND11_MODULE(mcts_ext, m) {
    m.doc() = "C++ Batched MCTS Extension for DeepVisionElite using bitboards";

    py::class_<Searcher>(m, "Searcher")
        .def(py::init<>())
        .def("set_root", &Searcher::set_root)
        .def("get_leaf_nodes", &Searcher::get_leaf_nodes)
        .def("expand_and_backprop", &Searcher::expand_and_backprop)
        .def("get_best_move", &Searcher::get_best_move);
        
    m.def("board_to_tensor_elite", &board_to_tensor_elite, "Encode board to 32x8x8 tensor array");
}
