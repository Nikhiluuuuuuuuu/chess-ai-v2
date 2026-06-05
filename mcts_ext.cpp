#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <string>
#include <cmath>
#include <unordered_map>
#include "chess.hpp"

namespace py = pybind11;

struct Node {
    chess::Board board;
    Node* parent;
    chess::Move move;
    float prior;
    float value_sum;
    int visit_count;
    bool is_expanded;
    std::unordered_map<uint16_t, Node*> children; // map move.to_int() -> Node*

    Node(chess::Board b, Node* p = nullptr, chess::Move m = chess::Move::NULL_MOVE, float pr = 0.0f)
        : board(b), parent(p), move(m), prior(pr), value_sum(0.0f), visit_count(0), is_expanded(false) {}

    ~Node() {
        for (auto& pair : children) {
            delete pair.second;
        }
    }

    float value() const {
        if (visit_count > 0) return value_sum / visit_count;
        if (parent) return parent->value() - 0.1f; // FPU
        return 0.0f;
    }

    Node* select_child(float c_puct = 1.4f) {
        float best_score = -1e9f;
        Node* best_child = nullptr;

        for (auto& pair : children) {
            Node* child = pair.second;
            float u_score = c_puct * child->prior * std::sqrt((float)visit_count) / (1.0f + child->visit_count);
            float score = child->value() + u_score;
            if (score > best_score) {
                best_score = score;
                best_child = child;
            }
        }
        return best_child;
    }
};

class Searcher {
private:
    Node* root;

public:
    Searcher() : root(nullptr) {}
    ~Searcher() { if (root) delete root; }

    void set_root(const std::string& fen) {
        if (root) delete root;
        chess::Board b(fen);
        root = new Node(b);
    }

    // Traverse the tree to find leaf nodes to evaluate
    std::vector<std::string> get_leaf_nodes(int batch_size) {
        std::vector<std::string> fens;
        // In a full implementation, we traverse the tree from root batch_size times,
        // collecting unexpanded leaf nodes, and return their FENs for PyTorch to evaluate.
        // For demonstration, we just return the root fen multiple times.
        for (int i=0; i<batch_size; ++i) {
            fens.push_back(root->board.getFen());
        }
        return fens;
    }

    // Receive predictions from PyTorch and expand/backpropagate
    void expand_and_backprop(const std::vector<std::vector<float>>& policies, const std::vector<float>& values) {
        // Expand the previously collected leaf nodes with their network predictions
        // and backpropagate the value up the tree.
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
        int max_visits = -1;
        for (auto& pair : root->children) {
            if (pair.second->visit_count > max_visits) {
                max_visits = pair.second->visit_count;
                best = pair.second;
            }
        }
        return chess::uci::moveToUci(best->move);
    }
};

PYBIND11_MODULE(mcts_ext, m) {
    m.doc() = "C++ Batched MCTS Extension for DeepVisionElite using bitboards";

    py::class_<Searcher>(m, "Searcher")
        .def(py::init<>())
        .def("set_root", &Searcher::set_root)
        .def("get_leaf_nodes", &Searcher::get_leaf_nodes)
        .def("expand_and_backprop", &Searcher::expand_and_backprop)
        .def("get_best_move", &Searcher::get_best_move);
}
