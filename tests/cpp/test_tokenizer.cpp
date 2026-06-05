#include "cppBpe/tokenizer.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <functional>
#include <iostream>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>
#include "absl/container/flat_hash_map.h"

using namespace cppBpe;

namespace
{
    int g_passed = 0;
    int g_failed = 0;
    bool g_current_failed = false;
    std::string g_current_name;

    void begin_test(const std::string& name)
    {
        g_current_name = name;
        g_current_failed = false;
    }

    void end_test()
    {
        if (g_current_failed)
        {
            std::cerr << "[FAILED]: " << g_current_name << std::endl;
            ++g_failed;
        }
        else
        {
            std::cout << "[PASSED]: " << g_current_name << std::endl;
            ++g_passed;
        }
    }

#define TEST(name) \
    do { \
        begin_test(#name); \
        name(); \
        end_test(); \
    } while (false)

#define CHECK(expr) \
    do { \
        if (!(expr)) \
        { \
            std::cerr << "  ASSERTION FAILED at " << __FILE__ << ":" << __LINE__ << ": " << #expr << std::endl; \
            g_current_failed = true; \
        } \
    } while (false)

#define CHECK_EQ(a, b) CHECK((a) == (b))
#define CHECK_NE(a, b) CHECK((a) != (b))

#define CHECK_THROW(expr) \
    do { \
        bool threw = false; \
        try { (expr); } \
        catch (...) { threw = true; } \
        if (!threw) \
        { \
            std::cerr << "  ASSERTION FAILED at " << __FILE__ << ":" << __LINE__ << ": expected exception from " << #expr << std::endl; \
            g_current_failed = true; \
        } \
    } while (false)


    Tokenizer make_tok_with_pattern(const std::string& pat, absl::flat_hash_map<Pair, TokenId, PairHash> merges = {})
    {
        Tokenizer tok(pat);
        tok.merges_ = std::move(merges);
        return tok;
    }

    void test_word_pairs()
    {
        const Word w({ 1, 2, 3, 4 });
        const std::vector<Pair> expected = { {1,2}, {2,3}, {3,4} };
        std::vector<Pair> got;
        for (std::size_t i = 0; i + 1 < w.ids.size(); ++i)
        {
            got.push_back({ w.ids[i], w.ids[i+1] });
        }
        CHECK_EQ(got, expected);
    }

    void test_word_pairs_empty()
    {
        const Word w({});
        CHECK(w.ids.size() < 2);
    }

    void test_word_pairs_single()
    {
        const Word w({ 42 });
        CHECK(w.ids.size() < 2);
    }

    void test_word_merge_pair()
    {
        // [1,2,3,1,2] merge (1,2)->99 -> [99,3,99]
        Word w({ 1, 2, 3, 1, 2 });
        w.merge_pair({ 1, 2 }, 99);
        CHECK_EQ(w.ids, (std::vector<TokenId>{ 99, 3, 99 }));
    }

    void test_word_merge_pair_adjacent()
    {
        // [1,2,1,2,1,2] -> [99,99,99]
        Word w({ 1, 2, 1, 2, 1, 2 });
        w.merge_pair({ 1, 2 }, 99);
        CHECK_EQ(w.ids, (std::vector<TokenId>{ 99, 99, 99 }));
    }

    void test_word_merge_no_match()
    {
        Word w({ 1, 2, 3 });
        auto deltas = w.merge_pair({ 4, 5 }, 99);
        CHECK_EQ(w.ids, (std::vector<TokenId>{ 1, 2, 3 }));
        const bool ok = deltas.empty() ||
                  std::ranges::all_of(deltas,
                                      [](const auto& kv) { return kv.second == 0; });
        CHECK(ok);
    }

    void test_tokenizer_new()
    {
        const Tokenizer tok;
        CHECK(tok.merges_.empty());
        CHECK(tok.get_patterns().empty());
    }

    void test_encode_with_pattern_no_merges()
    {
        // No merges, simple \w+ pattern -> raw byte values
        const auto tok = make_tok_with_pattern(R"(\w+)");
        const auto ids = tok.encode("hi");
        CHECK_EQ(ids, (std::vector<TokenId>{ 104, 105 }));
    }

    void test_encode_with_merges()
    {
        const auto tok = make_tok_with_pattern(R"(\w+)", { { {104, 105}, 256 } });

        const auto ids = tok.encode("hi");
        CHECK_EQ(ids, (std::vector<TokenId>{ 256 }));

        const auto ids2 = tok.encode("hip");
        CHECK_EQ(ids2, (std::vector<TokenId>{ 256, 112 }));
    }

    void test_get_mergeable_ranks_empty()
    {
        const Tokenizer tok;
        const auto ranks = tok.get_mergeable_ranks();
        CHECK_EQ(ranks.size(), static_cast<size_t>(256));
        // first: [0] -> 0
        CHECK_EQ(ranks[0].second, 0u);
        CHECK_EQ(ranks[0].first, (std::vector<uint8_t>{ 0 }));
        // last: [255] -> 255
        CHECK_EQ(ranks[255].second, 255u);
        CHECK_EQ(ranks[255].first, (std::vector<uint8_t>{ 255 }));
    }

    void test_get_mergeable_ranks_with_merge()
    {
        const auto tok = make_tok_with_pattern("", { { {65, 66}, 256 } });
        const auto ranks = tok.get_mergeable_ranks();
        CHECK_EQ(ranks.size(), static_cast<size_t>(257));
        CHECK_EQ(ranks[256].first, (std::vector<uint8_t>{ 65, 66 }));
        CHECK_EQ(ranks[256].second, 256u);
    }

    void test_count_pairs_parallel()
    {
        const std::vector<Word> words  = { Word({1,2,3}), Word({1,2,4}) };
        const std::vector<int32_t> counts = { 1, 2 };

        PairCounts    pc;
        WhereToUpdate wu;
        count_pairs_parallel(words, counts, pc, wu);

        // (1,2): 1*1 + 1*2 = 3
        CHECK_EQ(pc.at({1,2}), 3);
        // (2,3): 1
        CHECK_EQ(pc.at({2,3}), 1);
        // (2,4): 2
        CHECK_EQ(pc.at({2,4}), 2);

        const auto& v = wu.at({1,2});
        CHECK(std::find(v.begin(), v.end(), 0) != v.end());
        CHECK(std::find(v.begin(), v.end(), 1) != v.end());
    }

    void test_train_core_incremental()
    {
        Tokenizer tok(R"(\w+)");

        std::vector<std::string> texts(10, "ab");
        texts.insert(texts.end(), 5, "cd");
        tok.train(texts, 257);

        CHECK_EQ(tok.merges_.size(), static_cast<size_t>(1));
        CHECK(tok.merges_.count({97,98}));
        CHECK_EQ(tok.merges_.at({97,98}), 256u);
    }

    void test_default_vocab_size()
    {
        Tokenizer tok;
        CHECK_EQ(tok.vocab_size(), 256u);
        tok.merges_[{97,98}] = 256;
        CHECK_EQ(tok.vocab_size(), 257u);
    }

    void test_word_merge_overlapping_odd()
    {
        // "aaa" = [97,97,97] -> [256, 97]
        Word w({ 97, 97, 97 });
        w.merge_pair({ 97, 97 }, 256);
        CHECK_EQ(w.ids, (std::vector<TokenId>{ 256, 97 }));
    }

    void test_word_merge_overlapping_even()
    {
        // "aaaa" = [97,97,97,97] -> [256,256]
        Word w({ 97, 97, 97, 97 });
        w.merge_pair({ 97, 97 }, 256);
        CHECK_EQ(w.ids, (std::vector<TokenId>{ 256, 256 }));
    }

    void test_word_merge_multiple_occurrences()
    {
        Word w({ 1, 2, 99, 1, 2 });
        auto deltas = w.merge_pair({ 1, 2 }, 256);
        CHECK_EQ(w.ids, (std::vector<TokenId>{ 256, 99, 256 }));

        // (1,2) should have aggregate delta of -2
        int32_t ab_delta = 0;
        for (auto& [p, d] : deltas)
        {
            if (p == Pair{1, 2})
            {
                ab_delta += d;
            }
        }
        CHECK_EQ(ab_delta, -2);
    }

    void test_encode_chained_merges()
    {
        //std::unordered_map<Pair, TokenId, PairHash> m;
        absl::flat_hash_map<Pair, TokenId, PairHash> m;
        m[{97,97}]  = 256; // 'aa' -> 256 (learned first)
        m[{256,97}] = 257; // 'aa'+'a' -> 257 (learned second)
        const auto tok = make_tok_with_pattern(R"(\w+)", m);

        // "aaa" -> [257]
        CHECK_EQ(tok.encode("aaa"),  (std::vector<TokenId>{ 257 }));
        // "aaaa" -> [256, 256]
        CHECK_EQ(tok.encode("aaaa"), (std::vector<TokenId>{ 256, 256 }));
        // "aaaaa" -> [256, 257]
        CHECK_EQ(tok.encode("aaaaa"), (std::vector<TokenId>{ 256, 257 }));
    }

    void test_encode_decode_roundtrip_simple()
    {
        const auto tok = make_tok_with_pattern(R"(\w+|\s+)", { { {104,105}, 256 } });
        const auto ids = tok.encode("hi");
        const auto decoded = tok.decode(ids);
        CHECK_EQ(decoded, "hi");
    }

    void test_encode_decode_roundtrip_with_spaces()
    {
        //std::unordered_map<Pair, TokenId, PairHash> m;
        absl::flat_hash_map<Pair, TokenId, PairHash> m;
        m[{104,101}] = 256; // 'he'
        m[{108,108}] = 257; // 'll'
        m[{256,257}] = 258; // 'hell'
        const auto tok = make_tok_with_pattern(R"(\w+|\s+)", m);

        const std::string text = "hello world";
        const auto ids  = tok.encode(text);
        const auto decoded = tok.decode(ids);
        CHECK_EQ(decoded, text);
    }

    void test_decode_byte_level()
    {
        const Tokenizer tok;
        // [104, 105] = "hi"
        const auto decoded = tok.decode({ 104, 105 });
        CHECK_EQ(decoded, "hi");
    }

    void test_decode_invalid_token()
    {
        const Tokenizer tok;
        CHECK_THROW(tok.decode({ 300 }));
    }

    void test_train_creates_chained_merges()
    {
        // "aaa" ×10 -> first merge (97,97)->256, second (256,97)->257
        Tokenizer tok(R"(\w+)");
        const std::vector<std::string> texts(10, "aaa");
        tok.train(texts, 258);

        CHECK_EQ(tok.merges_.size(), static_cast<size_t>(2));
        CHECK(tok.merges_.count({97,97}));
        CHECK_EQ(tok.merges_.at({97,97}), 256u);
        CHECK(tok.merges_.count({256,97}));
        CHECK_EQ(tok.merges_.at({256,97}), 257u);
    }

    void test_get_mergeable_ranks_chained()
    {
        //std::unordered_map<Pair, TokenId, PairHash> m;
        absl::flat_hash_map<Pair, TokenId, PairHash> m;
        m[{65,66}]  = 256; // "AB"
        m[{256,67}] = 257; // "ABC"
        const auto tok = make_tok_with_pattern("", m);

        const auto ranks = tok.get_mergeable_ranks();
        CHECK_EQ(ranks.size(), static_cast<size_t>(258));
        CHECK_EQ(ranks[256].first,  (std::vector<uint8_t>{ 65, 66 }));
        CHECK_EQ(ranks[256].second, 256u);
        CHECK_EQ(ranks[257].first,  (std::vector<uint8_t>{ 65, 66, 67 }));
        CHECK_EQ(ranks[257].second, 257u);
    }

    void test_encode_empty_string()
    {
        const auto tok = make_tok_with_pattern(R"(\w+)");
        CHECK(tok.encode("").empty());
    }

    void test_encode_no_matches()
    {
        const auto tok = make_tok_with_pattern(R"(\w+)");
        CHECK(tok.encode("   ").empty());
    }

    void test_decode_empty()
    {
        const Tokenizer tok;
        CHECK_EQ(tok.decode({}), "");
    }

    void test_word_merge_deltas_correctness()
    {
        // Word: [1,2,3,1,2] merge (1,2)->99
        // Expected aggregate deltas: (1,2)=-2, (2,3)=-1, (3,1)=-1, (99,3)=+1, (3,99)=+1
        Word w({ 1, 2, 3, 1, 2 });
        auto deltas = w.merge_pair({ 1, 2 }, 99);

        std::unordered_map<Pair, int32_t, PairHash> dm;
        for (auto& [p, d] : deltas)
        {
            dm[p] += d;
        }

        CHECK_EQ(dm.at({1,2}), -2);
        CHECK_EQ(dm.at({2,3}), -1);
        CHECK_EQ(dm.at({3,1}), -1);
        CHECK_EQ(dm.at({99,3}), +1);
        CHECK_EQ(dm.at({3,99}), +1);
    }

    void test_count_pairs_parallel_empty()
    {
        PairCounts pc;
        WhereToUpdate wu;
        count_pairs_parallel({}, {}, pc, wu);
        CHECK(pc.empty());
        CHECK(wu.empty());
    }

    void test_count_pairs_parallel_zero_count()
    {
        const std::vector<Word> words = { Word({1,2,3}) };
        const std::vector<int32_t> counts = { 0 };
        PairCounts pc;
        WhereToUpdate wu;
        count_pairs_parallel(words, counts, pc, wu);
        CHECK(pc.empty());
    }

    void test_batch_encode_parallel()
    {
        const auto tok = make_tok_with_pattern(R"(\w+|\s+)", { { {104,105}, 256 } });
        const auto results = tok.batch_encode({ "hi", "hip", "hi" });
        CHECK_EQ(results.size(), static_cast<size_t>(3));
        CHECK_EQ(results[0], (std::vector<TokenId>{ 256 }));
        CHECK_EQ(results[1], (std::vector<TokenId>{ 256, 112 }));
        CHECK_EQ(results[2], (std::vector<TokenId>{ 256 }));
    }

    void test_unicode_pattern_gpt4()
    {
        // The GPT-4 pattern must match Unicode letters correctly.
        // "café" should split into chunks that include 'é' (U+00E9, 2 bytes UTF-8).
        const Tokenizer tok{std::string(GPT4_PATTERN)};
        const auto ids = tok.encode("café");
        // Roundtrip must work
        const auto decoded = tok.decode(ids);
        CHECK_EQ(decoded, "café");
    }
}

int main()
{
    try
    {
        TEST(test_word_pairs);
        TEST(test_word_pairs_empty);
        TEST(test_word_pairs_single);
        TEST(test_word_merge_pair);
        TEST(test_word_merge_pair_adjacent);
        TEST(test_word_merge_no_match);
        TEST(test_tokenizer_new);
        TEST(test_encode_with_pattern_no_merges);
        TEST(test_encode_with_merges);
        TEST(test_get_mergeable_ranks_empty);
        TEST(test_get_mergeable_ranks_with_merge);
        TEST(test_count_pairs_parallel);
        TEST(test_train_core_incremental);
        TEST(test_default_vocab_size);
        TEST(test_word_merge_overlapping_odd);
        TEST(test_word_merge_overlapping_even);
        TEST(test_word_merge_multiple_occurrences);
        TEST(test_encode_chained_merges);
        TEST(test_encode_decode_roundtrip_simple);
        TEST(test_encode_decode_roundtrip_with_spaces);
        TEST(test_decode_byte_level);
        TEST(test_decode_invalid_token);
        TEST(test_train_creates_chained_merges);
        TEST(test_get_mergeable_ranks_chained);
        TEST(test_encode_empty_string);
        TEST(test_encode_no_matches);
        TEST(test_decode_empty);
        TEST(test_word_merge_deltas_correctness);
        TEST(test_count_pairs_parallel_empty);
        TEST(test_count_pairs_parallel_zero_count);
        TEST(test_batch_encode_parallel);
        TEST(test_unicode_pattern_gpt4);
    }
    catch (const std::exception& ex)
    {
        std::cerr << "UNEXPECTED EXCEPTION: " << ex.what() << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << "UNEXPECTED UNKNOWN EXCEPTION" << std::endl;
        return 1;
    }

    std::cout << "Tests passed: " << g_passed << ", failed: " << g_failed << std::endl;
    return g_failed == 0 ? 0 : 1;
}