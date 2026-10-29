// Exact tropical scores for word sequences in a topologically ordered lattice.
// The acoustic lattice has already been pruned by the ordinary decoder. This
// operation only queries membership and does not change any path or weight.
#include <algorithm>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

extern "C" int lattice_trie_scores(
    int64_t num_states, int64_t num_arcs, const int32_t* arcs,
    const float* scores, const int32_t* word_splits, const int32_t* words,
    int64_t num_trie_nodes, int64_t num_trie_arcs, const int32_t* trie_arcs,
    int64_t num_cases, const int32_t* terminals, double* output) {
  try {
    if (num_states < 1 || num_trie_nodes < 1) return 1;
    std::vector<std::unordered_map<int32_t, int32_t>> trie(num_trie_nodes);
    for (int64_t i = 0; i < num_trie_arcs; ++i) {
      auto source = trie_arcs[3 * i], target = trie_arcs[3 * i + 1];
      if (source < 0 || source >= num_trie_nodes || target < 0 || target >= num_trie_nodes) return 2;
      trie[source][trie_arcs[3 * i + 2]] = target;
    }
    std::vector<std::unordered_map<int32_t, double>> forward(num_states);
    forward[0][0] = 0;
    int32_t previous_source = -1;
    for (int64_t i = 0; i < num_arcs; ++i) {
      auto source = arcs[4 * i], target = arcs[4 * i + 1];
      if (source < previous_source || source < 0 || target <= source || target >= num_states) return 3;
      if (previous_source >= 0 && previous_source != source) {
        std::unordered_map<int32_t, double>().swap(forward[previous_source]);
      }
      previous_source = source;
      for (const auto& entry : forward[source]) {
        int32_t node = entry.first;
        bool compatible = true;
        for (int32_t j = word_splits[i]; j < word_splits[i + 1]; ++j) {
          if (words[j] <= 0) continue;
          auto found = trie[node].find(words[j]);
          if (found == trie[node].end()) { compatible = false; break; }
          node = found->second;
        }
        if (!compatible) continue;
        double score = entry.second + scores[i];
        auto found = forward[target].find(node);
        if (found == forward[target].end()) forward[target][node] = score;
        else found->second = std::max(found->second, score);
      }
    }
    for (int64_t i = 0; i < num_cases; ++i) {
      auto found = forward.back().find(terminals[i]);
      output[i] = found == forward.back().end()
          ? -std::numeric_limits<double>::infinity() : found->second;
    }
    return 0;
  } catch (...) { return 4; }
}
