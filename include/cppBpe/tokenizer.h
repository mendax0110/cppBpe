#pragma once

#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <optional>
#include <ranges>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

struct pcre2_real_code_8;
struct pcre2_real_match_data_8;

namespace cppBpe
{
    using TokenId = uint32_t;

    struct Pair
    {
        TokenId first;
        TokenId second;

        bool operator==(const Pair& o) const noexcept
        {
            return first == o.first && second == o.second;
        }
    };
}

namespace cppBpe
{
    struct Word
    {
        std::vector<TokenId> ids;
        explicit Word(std::vector<TokenId> ids) noexcept : ids(std::move(ids))
        {
        }

        std::vector<std::pair<Pair, int32_t>> merge_pair(Pair pair, TokenId new_id);
    };

    struct PairHash
    {
        size_t operator()(const Pair p) const noexcept
        {
            const uint64_t combined = (static_cast<uint64_t>(p.first) << 32) | p.second;
            uint64_t h = combined ^ (combined >> 30);
            h *= 0xbf58476d1ce4e5b9ULL;
            h ^= h >> 27;
            h *= 0x94d049bb133111ebULL;
            h ^= h >> 31;
            return h;
        }
    };

    using PairCounts = std::unordered_map<Pair, int32_t, PairHash>;
    using WhereToUpdate = std::unordered_map<Pair, std::vector<size_t>, PairHash>;

    void count_pairs_parallel(const std::vector<Word>& words, const std::vector<int32_t>& counts, PairCounts& out_pair_counts, WhereToUpdate& out_where);

    class CompilePattern
    {
    public:
        explicit CompilePattern(std::string  pattern);
        ~CompilePattern();

        CompilePattern(const CompilePattern&);
        CompilePattern& operator=(const CompilePattern&);
        CompilePattern(CompilePattern&&) noexcept;
        CompilePattern& operator=(CompilePattern&&) noexcept;

        [[nodiscard]] std::vector<std::string_view> find_all(std::string_view text) const;
        [[nodiscard]] const std::string& pattern_str() const noexcept { return pattern_; }

        template<typename Fn>
        void for_each_match(std::string_view text, Fn&& callback) const;

    private:
        std::string pattern_;
        pcre2_real_code_8* code_ = nullptr;
        bool jit_ready_ = false;

        void compile();
        void free_code() noexcept;
    };

    constexpr std::string_view GPT4_PATTERN = R"('(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+)";

    class Tokenizer
    {
    public:
        Tokenizer();
        explicit Tokenizer(std::string pattern);

        template<typename InputIt>
        void train_from_iterator(InputIt begin, InputIt end, uint32_t vocab_size, size_t buffer_size = 8192, const std::optional<std::string>& pattern = std::nullopt);
        void train(const std::vector<std::string>& texts, uint32_t vocab_size, const std::optional<std::string> &pattern = std::nullopt);

        std::vector<TokenId> encode(std::string_view text) const;
        std::vector<std::vector<TokenId>> batch_encode(const std::vector<std::string>& texts) const;
        std::vector<TokenId> encode_chunk(std::string_view chunk) const;
        void encode_chuck_into(std::string_view chunk, std::vector<TokenId>& out) const;
        std::string decode(const std::vector<TokenId>& ids) const;

        std::unordered_map<Pair, TokenId, PairHash> merges_;
        uint32_t vocab_size() const noexcept
        {
            return 256U + static_cast<uint32_t>(merges_.size());
        }

        const std::string& get_patterns() const noexcept { return pattern_.pattern_str(); }
        std::vector<std::pair<std::vector<uint8_t>, uint32_t>> get_mergeable_ranks() const;

    private:
        CompilePattern pattern_;
        mutable std::vector<std::vector<uint8_t>> vocab_cache_;
        mutable bool vocab_dirty_ = true;

        const std::vector<std::vector<uint8_t>>& cached_vocab() const;
        void train_core_incremental(std::vector<Word> words, const std::vector<int32_t> &counts, uint32_t vocab_size);
        std::vector<std::vector<uint8_t>> build_vocab() const;

        //mutable std::unordered_map<uint64_t, TokenId> encode_map_;
        mutable std::vector<std::pair<uint64_t, TokenId>> encode_map_;
        mutable bool encode_map_dirty_ = true;

        //const std::unordered_map<uint64_t, TokenId>& cached_encode_map() const;
        const std::vector<std::pair<uint64_t, TokenId>>& cached_encode_map() const;

        [[nodiscard]] TokenId find_merge(const uint64_t key) const
        {
            const auto& map = cached_encode_map();
            const auto it = std::lower_bound(map.begin(), map.end(), key, [](const auto& entry, uint64_t k)
            {
               return entry.first < k;
            });
            if (it != map.end() && it->first == key)
            {
                return it->second;
            }
            return UINT32_MAX;
        }

        std::vector<TokenId> encode_sequential(std::string_view text) const;
    };

    template<typename InputIt>
    void Tokenizer::train_from_iterator(InputIt begin, InputIt end, const uint32_t vocab_size, size_t /*buffer_size*/, const std::optional<std::string>& pattern)
    {
        pattern_ = CompilePattern(pattern.value_or(std::string(GPT4_PATTERN)));
        const std::vector<std::string> lines(begin, end);

        using LocalCounts = std::unordered_map<std::string, int32_t>;
        tbb::enumerable_thread_specific<LocalCounts> local_counts;

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, lines.size()),
            [&](const tbb::blocked_range<size_t>& r)
            {
                auto& counts = local_counts.local();
                for (size_t i = r.begin(); i != r.end(); ++i)
                {
                    const std::string& text = lines[i];
                    for (auto sv : pattern_.find_all(text))
                    {
                        ++counts[std::string(sv)];
                    }
                }
            });

        LocalCounts counts;
        for (auto& local : local_counts)
        {
            for (auto& [chunk, count] : local)
            {
                counts[chunk] += count;
            }
        }

        std::vector<Word> words;
        std::vector<int32_t> cvec;
        words.reserve(counts.size());
        cvec.reserve(counts.size());

        for (const auto &chunk: counts | std::views::keys)
        {
            std::vector<TokenId> byte_ids;
            byte_ids.reserve(chunk.size());
            for (const unsigned char c : chunk)
            {
                byte_ids.push_back(c);
            }
            words.emplace_back(std::move(byte_ids));
            cvec.push_back(counts[chunk]);
        }

        train_core_incremental(std::move(words), std::move(cvec), vocab_size);
    }
}
