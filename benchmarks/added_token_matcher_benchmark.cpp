#include <aho_corasick/aho_corasick.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct AddedToken {
  std::string content;
};

struct Match {
  std::size_t token_index = 0;
  std::size_t start = 0;
  std::size_t stop = 0;
};

std::optional<Match> find_next_legacy(
    const std::vector<AddedToken> & tokens,
    std::string_view text,
    std::size_t search_start) {
  std::optional<Match> best;
  for (std::size_t token_index = 0; token_index < tokens.size(); ++token_index) {
    const auto & token = tokens[token_index];
    if (token.content.empty()) {
      continue;
    }
    const auto found = text.find(token.content, search_start);
    if (found == std::string_view::npos) {
      continue;
    }
    const auto stop = found + token.content.size();
    const auto length = stop - found;
    const auto best_length = best ? best->stop - best->start : 0;
    if (!best || found < best->start ||
        (found == best->start && length > best_length)) {
      best = Match{token_index, found, stop};
    }
  }
  return best;
}

std::size_t count_legacy_splits(
    const std::vector<AddedToken> & tokens,
    const std::string & text) {
  std::size_t count = 0;
  std::size_t search_start = 0;
  while (search_start < text.size()) {
    const auto match = find_next_legacy(tokens, text, search_start);
    if (!match) {
      break;
    }
    ++count;
    search_start = match->stop;
  }
  return count;
}

class TrieMatcher {
 public:
  explicit TrieMatcher(const std::vector<AddedToken> & tokens) {
    token_indices_by_emit_.reserve(tokens.size());
    for (std::size_t token_index = 0; token_index < tokens.size(); ++token_index) {
      if (tokens[token_index].content.empty()) {
        continue;
      }
      trie_.insert(tokens[token_index].content);
      token_indices_by_emit_.push_back(token_index);
    }
    (void)trie_.parse_text(std::string{});
  }

  std::vector<Match> find_matches(const std::string & text) {
    std::vector<Match> matches;
    for (const auto & emit : trie_.parse_text(text)) {
      const auto emit_index = static_cast<std::size_t>(emit.get_index());
      if (emit_index >= token_indices_by_emit_.size()) {
        continue;
      }
      matches.push_back(Match{
          token_indices_by_emit_[emit_index],
          emit.get_start(),
          emit.get_end() + 1});
    }
    std::sort(matches.begin(), matches.end(), [](const Match & lhs, const Match & rhs) {
      if (lhs.start != rhs.start) {
        return lhs.start < rhs.start;
      }
      const auto lhs_length = lhs.stop - lhs.start;
      const auto rhs_length = rhs.stop - rhs.start;
      if (lhs_length != rhs_length) {
        return lhs_length > rhs_length;
      }
      return lhs.token_index < rhs.token_index;
    });
    return matches;
  }

 private:
  aho_corasick::trie trie_;
  std::vector<std::size_t> token_indices_by_emit_;
};

std::size_t count_trie_splits(TrieMatcher & matcher, const std::string & text) {
  const auto matches = matcher.find_matches(text);
  std::size_t count = 0;
  std::size_t search_start = 0;
  for (const auto & match : matches) {
    if (match.start < search_start) {
      continue;
    }
    ++count;
    search_start = match.stop;
  }
  return count;
}

std::string numbered_token(std::size_t index) {
  std::ostringstream out;
  out << "<extra_" << std::setw(4) << std::setfill('0') << index << ">";
  return out.str();
}

std::vector<AddedToken> make_tokens(std::size_t count) {
  std::vector<AddedToken> tokens;
  tokens.reserve(count + 4);
  tokens.push_back(AddedToken{"<m>"});
  tokens.push_back(AddedToken{"<mask>"});
  tokens.push_back(AddedToken{"<mask>ing"});
  tokens.push_back(AddedToken{"\xE4\xB8\x96\xE7\x95\x8C"});
  for (std::size_t index = tokens.size(); index < count; ++index) {
    tokens.push_back(AddedToken{numbered_token(index)});
  }
  return tokens;
}

std::string make_text(std::size_t repeats, std::size_t token_count) {
  std::string text;
  text.reserve(repeats * 48);
  for (std::size_t index = 0; index < repeats; ++index) {
    text += "hello ";
    text += numbered_token((index % std::max<std::size_t>(token_count, 8)));
    text += " <mask>ing ";
    text += "\xE4\xB8\x96\xE7\x95\x8C";
    text += " tail ";
  }
  return text;
}

template <typename Fn>
double time_ms(Fn fn) {
  const auto start = std::chrono::steady_clock::now();
  volatile std::size_t sink = fn();
  (void)sink;
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(end - start).count();
}

void run_case(std::size_t token_count, std::size_t repeats) {
  auto tokens = make_tokens(token_count);
  auto text = make_text(repeats, token_count);
  TrieMatcher matcher(tokens);

  const auto legacy_ms = time_ms([&] {
    std::size_t total = 0;
    for (std::size_t iteration = 0; iteration < 20; ++iteration) {
      total += count_legacy_splits(tokens, text);
    }
    return total;
  });
  const auto trie_ms = time_ms([&] {
    std::size_t total = 0;
    for (std::size_t iteration = 0; iteration < 20; ++iteration) {
      total += count_trie_splits(matcher, text);
    }
    return total;
  });

  std::cout << "tokens=" << token_count << " repeats=" << repeats
            << " legacy_ms=" << legacy_ms << " trie_ms=" << trie_ms
            << " speedup=" << (legacy_ms / trie_ms) << "x\n";
}

}  // namespace

int main() {
  run_case(16, 64);
  run_case(128, 128);
  run_case(1024, 256);
  return 0;
}
