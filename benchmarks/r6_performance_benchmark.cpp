#include "tokenizers_cpp/tokenizer.hpp"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace {

using json = nlohmann::json;
using Clock = std::chrono::steady_clock;

struct BenchmarkResult {
  std::string name;
  std::size_t iterations = 0;
  std::size_t input_bytes = 0;
  std::uint64_t checksum = 0;
  double elapsed_ms = 0.0;
};

std::filesystem::path write_temp_tokenizer_json(
    const std::string & name,
    const json & value) {
  const auto path = std::filesystem::temp_directory_path() / (name + ".json");
  std::ofstream output(path);
  output << value;
  return path;
}

json read_json(const std::filesystem::path & path) {
  std::ifstream input(path);
  json value;
  input >> value;
  return value;
}

tokenizers_cpp::Tokenizer load_tokenizer(const std::string & name, const json & value) {
  const auto path = write_temp_tokenizer_json(name, value);
  const auto tokenizer = tokenizers_cpp::Tokenizer::from_file(path);
  std::filesystem::remove(path);
  return tokenizer;
}

json tokenizer_json(json model, json pre_tokenizer = nullptr) {
  return json{
      {"version", "1.0"},
      {"truncation", nullptr},
      {"padding", nullptr},
      {"added_tokens", json::array()},
      {"normalizer", nullptr},
      {"pre_tokenizer", std::move(pre_tokenizer)},
      {"post_processor", nullptr},
      {"decoder", nullptr},
      {"model", std::move(model)},
  };
}

std::uint64_t checksum_encoding(const tokenizers_cpp::Encoding & encoding) {
  std::uint64_t value = encoding.ids.size();
  for (const auto id : encoding.ids) {
    value = value * 1315423911ULL + id;
  }
  for (const auto & offset : encoding.offsets) {
    value = value * 16777619ULL + offset.start;
    value = value * 16777619ULL + offset.end;
  }
  for (const auto & overflow : encoding.overflowing) {
    value = value * 1099511628211ULL + checksum_encoding(overflow);
  }
  return value;
}

std::uint64_t checksum_string(std::string_view text) {
  std::uint64_t value = 1469598103934665603ULL;
  for (const auto byte : text) {
    value ^= static_cast<unsigned char>(byte);
    value *= 1099511628211ULL;
  }
  return value;
}

std::uint64_t checksum_encodings(const std::vector<tokenizers_cpp::Encoding> & encodings) {
  std::uint64_t value = encodings.size();
  for (const auto & encoding : encodings) {
    value = value * 1315423911ULL + checksum_encoding(encoding);
  }
  return value;
}

std::uint64_t checksum_strings(const std::vector<std::string> & strings) {
  std::uint64_t value = strings.size();
  for (const auto & string : strings) {
    value = value * 1315423911ULL + checksum_string(string);
  }
  return value;
}

std::size_t input_bytes(const std::vector<std::string> & strings) {
  std::size_t total = 0;
  for (const auto & string : strings) {
    total += string.size();
  }
  return total;
}

std::size_t input_bytes(const std::vector<std::pair<std::string, std::string>> & pairs) {
  std::size_t total = 0;
  for (const auto & pair : pairs) {
    total += pair.first.size() + pair.second.size();
  }
  return total;
}

template <typename Fn>
BenchmarkResult run_case(
    std::string name,
    std::size_t iterations,
    std::size_t input_bytes,
    Fn fn) {
  std::uint64_t checksum = 0;
  const auto start = Clock::now();
  for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
    checksum += fn();
  }
  const auto end = Clock::now();
  return BenchmarkResult{
      std::move(name),
      iterations,
      input_bytes,
      checksum,
      std::chrono::duration<double, std::milli>(end - start).count()};
}

std::string numbered_token(std::size_t index) {
  std::ostringstream out;
  out << "<extra_" << std::setw(4) << std::setfill('0') << index << ">";
  return out.str();
}

tokenizers_cpp::Tokenizer make_added_token_tokenizer(std::size_t added_count) {
  auto tokenizer = load_tokenizer(
      "tokenizers_cpp_bench_added_tokens",
      tokenizer_json(
          {
              {"type", "WordLevel"},
              {"unk_token", "<unk>"},
              {"vocab", {{"<unk>", 0}, {"hello", 1}, {"tail", 2}}},
          },
          {{"type", "Whitespace"}}));

  std::vector<tokenizers_cpp::AddedToken> tokens;
  tokens.reserve(added_count + 4);
  tokens.push_back(tokenizers_cpp::AddedToken{"<m>"});
  tokens.push_back(tokenizers_cpp::AddedToken{"<mask>"});
  tokens.push_back(tokenizers_cpp::AddedToken{"<mask>ing"});
  tokens.push_back(tokenizers_cpp::AddedToken{"\xE4\xB8\x96\xE7\x95\x8C"});
  for (std::size_t index = tokens.size(); index < added_count; ++index) {
    tokens.push_back(tokenizers_cpp::AddedToken{numbered_token(index)});
  }
  (void)tokenizer.add_tokens(tokens);
  return tokenizer;
}

std::string make_added_token_text(std::size_t repeats, std::size_t added_count) {
  std::string text;
  text.reserve(repeats * 48);
  for (std::size_t index = 0; index < repeats; ++index) {
    text += "hello ";
    text += numbered_token(index % added_count);
    text += " <mask>ing ";
    text += "\xE4\xB8\x96\xE7\x95\x8C";
    text += " tail ";
  }
  return text;
}

json short_bpe_model() {
  return {
      {"type", "BPE"},
      {"vocab",
       {
           {"h", 0},
           {"e", 1},
           {"l", 2},
           {"o", 3},
           {"he", 4},
           {"hel", 5},
           {"hell", 6},
           {"hello", 7},
       }},
      {"merges",
       json::array({
           json::array({"h", "e"}),
           json::array({"he", "l"}),
           json::array({"hel", "l"}),
           json::array({"hell", "o"}),
       })},
  };
}

json power_bpe_model(std::size_t max_power) {
  json vocab = json::object();
  json merges = json::array();
  std::string token = "a";
  std::uint32_t id = 0;
  vocab[token] = id++;

  while (token.size() < max_power) {
    const auto left = token;
    const auto right = token;
    token += token;
    vocab[token] = id++;
    merges.push_back(json::array({left, right}));
  }

  return {
      {"type", "BPE"},
      {"vocab", std::move(vocab)},
      {"merges", std::move(merges)},
  };
}

json unigram_model() {
  return {
      {"type", "Unigram"},
      {"unk_id", 0},
      {"vocab",
       json::array({
           json::array({"<unk>", -100.0}),
           json::array({"a", 0.0}),
           json::array({"b", 0.0}),
           json::array({"ab", 2.0}),
       })},
  };
}

json wordpiece_model() {
  return {
      {"type", "WordPiece"},
      {"unk_token", "[UNK]"},
      {"continuing_subword_prefix", "##"},
      {"max_input_chars_per_word", 100},
      {"vocab",
       {
           {"[UNK]", 0},
           {"hello", 1},
           {"world", 2},
           {"token", 3},
           {"##izer", 4},
           {"##s", 5},
           {"benchmark", 6},
       }},
  };
}

std::string repeat_word(std::string_view word, std::size_t count) {
  std::string text;
  text.reserve((word.size() + 1) * count);
  for (std::size_t index = 0; index < count; ++index) {
    if (index > 0) {
      text.push_back(' ');
    }
    text.append(word);
  }
  return text;
}

std::vector<std::string> repeat_inputs(
    const std::vector<std::string> & base,
    std::size_t repeats) {
  std::vector<std::string> inputs;
  inputs.reserve(base.size() * repeats);
  for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
    inputs.insert(inputs.end(), base.begin(), base.end());
  }
  return inputs;
}

std::vector<std::pair<std::string, std::string>> repeat_pair_inputs(
    const std::vector<std::pair<std::string, std::string>> & base,
    std::size_t repeats) {
  std::vector<std::pair<std::string, std::string>> inputs;
  inputs.reserve(base.size() * repeats);
  for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
    inputs.insert(inputs.end(), base.begin(), base.end());
  }
  return inputs;
}

void print_result(const BenchmarkResult & result);

#ifdef TOKENIZERS_CPP_HF_TEST_DATA_DIR
tokenizers_cpp::Tokenizer load_gpt_style_sequence_tokenizer(
    const std::filesystem::path & data_dir) {
  auto value = read_json(data_dir / "tokenizer.json");
  value["pre_tokenizer"] = {
      {"type", "ByteLevel"},
      {"add_prefix_space", true},
      {"trim_offsets", true},
      {"use_regex", true},
  };
  value["decoder"] = {
      {"type", "ByteLevel"},
      {"add_prefix_space", true},
      {"trim_offsets", true},
      {"use_regex", true},
  };
  value["post_processor"] = {
      {"type", "Sequence"},
      {"processors",
       json::array({
           {
               {"type", "ByteLevel"},
               {"add_prefix_space", true},
               {"trim_offsets", true},
               {"use_regex", true},
           },
           {
               {"type", "TemplateProcessing"},
               {"single",
                json::array({
                    {{"SpecialToken", {{"id", "<s>"}, {"type_id", 0}}}},
                    {{"Sequence", {{"id", "A"}, {"type_id", 0}}}},
                })},
               {"pair",
                json::array({
                    {{"SpecialToken", {{"id", "<s>"}, {"type_id", 0}}}},
                    {{"Sequence", {{"id", "A"}, {"type_id", 0}}}},
                    {{"SpecialToken", {{"id", "<s>"}, {"type_id", 1}}}},
                    {{"Sequence", {{"id", "B"}, {"type_id", 1}}}},
                })},
               {"special_tokens",
                {
                    {"<s>",
                     {
                         {"id", "<s>"},
                         {"ids", json::array({0})},
                         {"tokens", json::array({"<s>"})},
                     }},
                }},
           },
       })},
  };

  return load_tokenizer("tokenizers_cpp_bench_gpt_sequence", value);
}

void run_real_tokenizer_matrix(const std::filesystem::path & data_dir) {
  {
    auto tokenizer = load_gpt_style_sequence_tokenizer(data_dir);
    const auto texts = repeat_inputs({"the a", "the", "a the", "the a the"}, 16);
    (void)tokenizer.encode_batch(texts, true);
    print_result(run_case("real_gpt_sequence_batch", 120, input_bytes(texts), [&] {
      const auto outputs = tokenizer.encode_batch(texts, true);
      return checksum_encodings(outputs) +
          checksum_strings(tokenizer.decode_batch(
              std::vector<std::vector<std::uint32_t>>{
                  outputs[0].ids,
                  outputs[1].ids,
                  outputs[2].ids,
                  outputs[3].ids,
              },
              true));
    }));
  }

  {
    auto tokenizer =
        tokenizers_cpp::Tokenizer::from_file(data_dir / "roberta.json");
    const auto pairs = repeat_pair_inputs(
        {{"Hello", "world"}, {"Hello", "test"}, {"Hello", "tokenizers"}},
        12);
    (void)tokenizer.encode_batch_pairs(pairs, true);
    print_result(run_case("real_roberta_pair_batch", 80, input_bytes(pairs), [&] {
      const auto outputs = tokenizer.encode_batch_pairs(pairs, true);
      return checksum_encodings(outputs) +
          checksum_strings(tokenizer.decode_batch(
              std::vector<std::vector<std::uint32_t>>{
                  outputs[0].ids,
                  outputs[1].ids,
                  outputs[2].ids,
              },
              true));
    }));
  }

  {
    auto tokenizer =
        tokenizers_cpp::Tokenizer::from_file(data_dir / "bert-wiki.json");
    tokenizer.with_wordpiece_decoder();
    const auto texts = repeat_inputs(
        {
            "Welcome to the \xF0\x9F\xA4\x97 Tokenizers library.",
            "Hello world",
            "This tokenizer benchmark checks batch decode.",
        },
        10);
    (void)tokenizer.encode_batch(texts, true);
    print_result(run_case("real_bert_batch_decode", 80, input_bytes(texts), [&] {
      const auto outputs = tokenizer.encode_batch(texts, true);
      return checksum_encodings(outputs) +
          checksum_strings(tokenizer.decode_batch(
              std::vector<std::vector<std::uint32_t>>{
                  outputs[0].ids,
                  outputs[1].ids,
                  outputs[2].ids,
              },
              true));
    }));
  }

  {
    auto tokenizer = tokenizers_cpp::Tokenizer::from_file(
        data_dir / "albert-base-v1-tokenizer.json");
    const auto texts = repeat_inputs(
        {"Hello world", "Hello world test", "SentencePiece unigram benchmark"},
        12);
    (void)tokenizer.encode_batch(texts, true);
    print_result(run_case("real_albert_batch_decode", 80, input_bytes(texts), [&] {
      const auto outputs = tokenizer.encode_batch(texts, true);
      return checksum_encodings(outputs) +
          checksum_strings(tokenizer.decode_batch(
              std::vector<std::vector<std::uint32_t>>{
                  outputs[0].ids,
                  outputs[1].ids,
                  outputs[2].ids,
              },
              true));
    }));
  }

  {
    auto tokenizer =
        tokenizers_cpp::Tokenizer::from_file(data_dir / "llama-3-tokenizer.json");
    const auto pairs = repeat_pair_inputs(
        {
            {"Hello", "world"},
            {"Hey! how is this token: ", "I'm you're they'll don't"},
            {"caf\xC3\xA9 \xE6\x9D\xB1\xE4\xBA\xAC", "abc 1234"},
        },
        10);
    (void)tokenizer.encode_batch_pairs(pairs, true);
    print_result(run_case("real_llama_pair_batch", 60, input_bytes(pairs), [&] {
      const auto outputs = tokenizer.encode_batch_pairs(pairs, true);
      return checksum_encodings(outputs) +
          checksum_strings(tokenizer.decode_batch(
              std::vector<std::vector<std::uint32_t>>{
                  outputs[0].ids,
                  outputs[1].ids,
                  outputs[2].ids,
              },
              true));
    }));
  }
}
#endif

void print_result(const BenchmarkResult & result) {
  const auto total_bytes =
      static_cast<double>(result.input_bytes * result.iterations);
  const auto seconds = result.elapsed_ms / 1000.0;
  const auto mib_per_s = seconds > 0.0
      ? total_bytes / (1024.0 * 1024.0) / seconds
      : 0.0;
  const auto encodes_per_s = seconds > 0.0
      ? static_cast<double>(result.iterations) / seconds
      : 0.0;

  std::cout << std::left << std::setw(34) << result.name
            << " iterations=" << std::setw(5) << result.iterations
            << " input_bytes=" << std::setw(8) << result.input_bytes
            << " elapsed_ms=" << std::fixed << std::setprecision(3)
            << std::setw(10) << result.elapsed_ms
            << " MiB/s=" << std::setw(10) << mib_per_s
            << " enc/s=" << std::setw(10) << encodes_per_s
            << " checksum=" << result.checksum << "\n";
}

}  // namespace

int main() {
  {
    constexpr std::size_t kAddedCount = 1024;
    auto tokenizer = make_added_token_tokenizer(kAddedCount);
    const auto text = make_added_token_text(256, kAddedCount);
    (void)tokenizer.encode(text, false);
    print_result(run_case("added_tokens_1024", 100, text.size(), [&] {
      return checksum_encoding(tokenizer.encode(text, false));
    }));
  }

  {
    const auto tokenizer = load_tokenizer(
        "tokenizers_cpp_bench_bpe_cache",
        tokenizer_json(short_bpe_model(), {{"type", "Whitespace"}}));
    const auto text = repeat_word("hello", 2048);
    (void)tokenizer.encode(text, false);
    print_result(run_case("bpe_cache_repeated_hello", 200, text.size(), [&] {
      return checksum_encoding(tokenizer.encode(text, false));
    }));
  }

  {
    const auto tokenizer = load_tokenizer(
        "tokenizers_cpp_bench_bpe_heap",
        tokenizer_json(power_bpe_model(512)));
    const std::string text(512, 'a');
    (void)tokenizer.encode(text, false);
    print_result(run_case("bpe_heap_long_piece_512", 500, text.size(), [&] {
      return checksum_encoding(tokenizer.encode(text, false));
    }));
  }

  {
    const auto tokenizer = load_tokenizer(
        "tokenizers_cpp_bench_unigram",
        tokenizer_json(unigram_model(), {{"type", "WhitespaceSplit"}}));
    const auto text = repeat_word("ab", 4096);
    (void)tokenizer.encode(text, false);
    print_result(run_case("unigram_trie_cache_ab", 200, text.size(), [&] {
      return checksum_encoding(tokenizer.encode(text, false));
    }));
  }

  {
    const auto tokenizer = load_tokenizer(
        "tokenizers_cpp_bench_wordpiece",
        tokenizer_json(wordpiece_model(), {{"type", "Whitespace"}}));
    const auto text = repeat_word("hello tokenizers world benchmark", 512);
    (void)tokenizer.encode(text, false);
    print_result(run_case("wordpiece_trie_repeated", 200, text.size(), [&] {
      return checksum_encoding(tokenizer.encode(text, false));
    }));
  }

  {
    const auto tokenizer = load_tokenizer(
        "tokenizers_cpp_bench_wordpiece_small_batch",
        tokenizer_json(wordpiece_model(), {{"type", "Whitespace"}}));
    const std::vector<std::string> texts{
        "hello world",
        "hello tokenizers",
        "tokenizers benchmark",
        "hello world benchmark",
    };
    (void)tokenizer.encode_batch(texts, false);
    print_result(run_case("wordpiece_small_batch", 1000, input_bytes(texts), [&] {
      return checksum_encodings(tokenizer.encode_batch(texts, false));
    }));
  }

#ifdef TOKENIZERS_CPP_HF_TEST_DATA_DIR
  run_real_tokenizer_matrix(
      std::filesystem::path(TOKENIZERS_CPP_HF_TEST_DATA_DIR));
#endif

  return 0;
}
