// Uses the exact upstream tokenizer implementation; no alternate vocabulary algorithm.
#include "utils/trie.h"
#include <iomanip>
int main(int argc, char** argv) {
  if (argc != 2) return 2;
  OptimizedTrieTokenizer tokenizer(argv[1]);
  if (!tokenizer.inited()) return 3;
  std::string hex;
  while (std::getline(std::cin, hex)) {
    if (hex.size() % 2) return 4;
    std::string input;
    for (size_t i = 0; i < hex.size(); i += 2) {
      if (!std::isxdigit(hex[i]) || !std::isxdigit(hex[i+1])) return 4;
      input.push_back(static_cast<char>(std::stoi(hex.substr(i, 2), nullptr, 16)));
    }
    auto ids = tokenizer.encode(input);
    if (tokenizer.decode(ids) != input) return 5;
    std::cout << '[';
    for (size_t i = 0; i < ids.size(); ++i) { if (i) std::cout << ','; std::cout << ids[i]; }
    std::cout << "]\n";
  }
}
