#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>
#include <tbb/parallel_reduce.h>
#include "cppBpe/tokenizer.h"
#include <algorithm>
#include <cassert>
#include <queue>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <numeric>

using namespace cppBpe;

std::vector<std::pair<Pair, int32_t>> Word::merge_pair(const Pair pair, const TokenId new_id)
{
    const TokenId a = pair.first;
    const TokenId b = pair.second;
    const size_t n = ids.size();

    if (n < 2)
    {
        return {};
    }

    std::vector<TokenId> out;
    out.reserve(n);

    std::vector<std::pair<Pair, int32_t>> deltas;
    deltas.reserve(6);

    size_t i = 0;
    while (i < n)
    {
        if (i + 1 < n && ids[i] == a && ids[i + 1] == b)
        {
            std::optional<TokenId> left = out.empty() ? std::optional<TokenId>{} : std::make_optional(out.back());
            std::optional<TokenId> right = (i + 2 < n) ? std::make_optional(ids[i + 2]) : std::optional<TokenId>{};

            if (left)
            {
                deltas.emplace_back(Pair{*left, a}, -1);
                deltas.emplace_back(Pair{*left, new_id}, +1);
            }

            deltas.emplace_back(Pair{a, b}, -1);

            if (right)
            {
                deltas.emplace_back(Pair{b, *right}, -1);
                deltas.emplace_back(Pair{new_id, *right}, +1);
            }

            out.push_back(new_id);
            i += 2;
        }
        else
        {
            out.push_back(ids[i]);
            ++i;
        }
    }

    ids = std::move(out);
    return deltas;
}

CompilePattern::CompilePattern(std::string pattern)
    : pattern_(std::move(pattern))
{
    compile();
}

CompilePattern::~CompilePattern()
{
    free_code();
}

CompilePattern::CompilePattern(const CompilePattern& other)
    : pattern_(other.pattern_)
{
    compile();
}

CompilePattern& CompilePattern::operator=(const CompilePattern& other)
{
    if (this != &other)
    {
        free_code();
        pattern_ = other.pattern_;
        compile();
    }
    return *this;
}

CompilePattern::CompilePattern(CompilePattern&& other) noexcept
    : pattern_(std::move(other.pattern_))
    , code_(other.code_)
{
    other.code_ = nullptr;
}

CompilePattern& CompilePattern::operator=(CompilePattern&& other) noexcept
{
    if (this != &other)
    {
        free_code();
        pattern_ = std::move(other.pattern_);
        code_ = other.code_;
        other.code_ = nullptr;
    }
    return *this;
}

void CompilePattern::compile()
{
    if (pattern_.empty())
    {
        code_ = nullptr;
        jit_ready_ = false;
        return;
    }

    int errcode = 0;
    PCRE2_SIZE erroffset = 0;

    code_ = pcre2_compile_8(
        reinterpret_cast<PCRE2_SPTR8>(pattern_.c_str()),
        PCRE2_ZERO_TERMINATED,
        PCRE2_UTF | PCRE2_UCP,
        &errcode,
        &erroffset,
        nullptr
    );
    if (!code_)
    {
        PCRE2_UCHAR8 errbuffer[256];
        pcre2_get_error_message_8(errcode, errbuffer, sizeof(errbuffer));
        throw std::runtime_error(
            std::string("invalid regex pattern at offest ") +
            std::to_string(erroffset) + ": " + reinterpret_cast<char*>(errbuffer));
    }

    const int jit_rc = pcre2_jit_compile_8(code_, PCRE2_JIT_COMPLETE);
    jit_ready_ = (jit_rc == 0);

    static bool warned = false;
    if (!warned)
    {
        int jit_config = 0;
        const int cfg_rc = pcre2_config_8(PCRE2_CONFIG_JIT, &jit_config);
        std::cerr << "[cppBpe] PCRE2 JIT config=" << jit_config
                  << " config_rc=" << cfg_rc
                  << " jit_compile_rc=" << jit_rc
                  << (jit_ready_ ? " (enabled)" : " (disabled)")
                  << "\n";

        if (!jit_ready_)
        {
            std::cerr << "[cppBpe] PCRE2 JIT unavailable; falling back to pcre2_match_8.\n";
        }
        warned = true;
    }
}

void CompilePattern::free_code() noexcept
{
    if (code_)
    {
        pcre2_code_free_8(code_);
        code_ = nullptr;
    }

    for (auto* md : tls_match_data_)
    {
        if (md)
        {
            pcre2_match_data_free_8(md);
        }
    }

    jit_ready_ = false;
}

std::vector<std::string_view> CompilePattern::find_all(const std::string_view text) const
{
    std::vector<std::string_view> results;

    if (!code_ || text.empty())
    {
        return results;
    }

    pcre2_match_data_8* md = pcre2_match_data_create_from_pattern_8(code_, nullptr);
    if (!md)
    {
        throw std::runtime_error("failed to create pcre2 match data");
    }

    const auto subject = reinterpret_cast<PCRE2_SPTR8>(text.data());
    const PCRE2_SIZE length = text.size();
    PCRE2_SIZE offset = 0;

    while (offset <= length)
    {
        int rc = 0;
        if (jit_ready_)
        {
            rc = pcre2_jit_match_8(code_, subject, length, offset, 0, md, nullptr);
            if (rc == PCRE2_ERROR_JIT_BADOPTION || rc == PCRE2_ERROR_JIT_STACKLIMIT)
            {
                rc = pcre2_match_8(code_, subject, length, offset, 0, md, nullptr);
            }
        }
        else
        {
            rc = pcre2_match_8(code_, subject, length, offset, 0, md, nullptr);
        }

        if (rc == PCRE2_ERROR_NOMATCH)
        {
            break;
        }

        if (rc < 0)
        {
            break;
        }

        const PCRE2_SIZE* ov = pcre2_get_ovector_pointer_8(md);
        const PCRE2_SIZE start = ov[0];
        const PCRE2_SIZE end = ov[1];

        if (end == start)
        {
            ++offset;
            continue;
        }

        results.push_back(text.substr(start, end - start));
        offset = end;
    }

    pcre2_match_data_free_8(md);
    return results;
}

template<typename Fn>
void CompilePattern::for_each_match(const std::string_view text, Fn&& callback) const
{
    if (!code_ || text.empty())
    {
        return;
    }

    pcre2_match_data_8* md = tls_match_data_.local();
    if (!md)
    {
        md = pcre2_match_data_create_from_pattern_8(code_, nullptr);
    }

    const auto subject = reinterpret_cast<PCRE2_SPTR8>(text.data());
    const PCRE2_SIZE length = text.size();
    PCRE2_SIZE offset = 0;

    while (offset <= length)
    {
        int rc = 0;
        if (jit_ready_)
        {
            rc = pcre2_jit_match_8(code_, subject, length, offset, 0, md, nullptr);
            if (rc == PCRE2_ERROR_JIT_BADOPTION || rc == PCRE2_ERROR_JIT_STACKLIMIT)
            {
                rc = pcre2_match_8(code_, subject, length, offset, 0, md, nullptr);
            }
        }
        else
        {
            rc = pcre2_match_8(code_, subject, length, offset, 0, md, nullptr);
        }

        if (rc == PCRE2_ERROR_NOMATCH)
        {
            break;
        }

        if (rc < 0)
        {
            break;
        }

        const PCRE2_SIZE* ov = pcre2_get_ovector_pointer_8(md);
        const PCRE2_SIZE start = ov[0];
        const PCRE2_SIZE end = ov[1];

        if (end == start)
        {
            ++offset;
            continue;
        }

        callback(text.substr(start, end - start));
        offset = end;
    }
}

void cppBpe::count_pairs_parallel(const std::vector<Word>& words, const std::vector<int32_t>& counts, PairCounts& out_pair_counts, WhereToUpdate& out_where)
{
    if (words.empty())
    {
        return;
    }

    struct Accum
    {
        PairCounts pc;
        WhereToUpdate wu;
    };

    Accum result = tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0, words.size()),
        Accum{},
        [&](const tbb::blocked_range<size_t>& r, Accum acc) -> Accum
        {
            for (size_t i = r.begin(); i != r.end(); ++i)
            {
                const int32_t c = counts[i];
                if (c == 0 || words[i].ids.size() < 2)
                {
                    continue;
                }
                const auto& ids = words[i].ids;
                for (size_t j = 0; j + 1 < ids.size(); ++j)
                {
                    Pair p{ids[j], ids[j + 1]};
                    acc.pc[p] += c;
                    acc.wu[p].push_back(i);
                }
            }
            return acc;
        },
        [](Accum left, Accum right) -> Accum
        {
            for (auto& [k, v] : right.pc)
            {
                left.pc[k] += v;
            }
            for (auto& [k, s] : right.wu)
            {
                auto& dest = left.wu[k];
                left.wu[k].insert(dest.end(), s.begin(), s.end());
            }
            return left;
        });

    out_pair_counts = std::move(result.pc);
    out_where = std::move(result.wu);
}

Tokenizer::Tokenizer()
    : pattern_(std::string(""))
{
}

Tokenizer::Tokenizer(std::string pattern)
    : pattern_(std::move(pattern))
{
}

namespace
{
    struct MergeJob
    {
        Pair pair;
        uint64_t count = 0;
        std::vector<size_t> pos;

        bool operator<(const MergeJob& o) const noexcept
        {
            if (count != o.count)
            {
                return count < o.count;
            }

            if (pair.first != o.pair.first)
            {
                return pair.first > o.pair.first;
            }

            return pair.second > o.pair.second;
        }
    };
}

void Tokenizer::train_core_incremental(std::vector<Word> words, const std::vector<int32_t>& counts, const uint32_t vocab_size)
{
    assert(vocab_size >= 256 && "vocab_size must be at least 256");

    const uint32_t num_merges = vocab_size - 256;
    merges_.clear();
    merges_.reserve(num_merges);

    PairCounts pair_counts;
    WhereToUpdate where_to_update;
    count_pairs_parallel(words, counts, pair_counts, where_to_update);

    std::priority_queue<MergeJob> heap;

    for (auto& [pair, pos_set] : where_to_update)
    {
        int32_t c = 0;
        if (auto it = pair_counts.find(pair); it != pair_counts.end())
        {
            c = it->second;
        }

        if (c > 0)
        {
            heap.push(MergeJob{pair, static_cast<uint64_t>(c), std::move(pos_set)});
        }
    }

    where_to_update.clear();

    uint32_t merges_done = 0;

    while (merges_done < num_merges && !heap.empty())
    {
        MergeJob top = heap.top();
        heap.pop();

        int32_t current = 0;
        if (auto it = pair_counts.find(top.pair); it != pair_counts.end())
        {
            current = it->second;
        }

        if (current <= 0)
        {
            continue;
        }

        if (static_cast<uint64_t>(current) != top.count)
        {
            top.count = static_cast<uint64_t>(current);
            heap.push(std::move(top));
            continue;
        }

        const TokenId new_id = 256U + merges_done;
        merges_[top.pair] = new_id;

        WhereToUpdate local_pos_updates;

        for (size_t word_idx : top.pos)
        {
            auto changes = words[word_idx].merge_pair(top.pair, new_id);
            for (auto& [p, delta] : changes)
            {
                const int32_t delta_total = delta * counts[word_idx];
                if (delta_total == 0)
                {
                    continue;
                }

                pair_counts[p] += delta_total;

                if (delta > 0)
                {
                    local_pos_updates[p].push_back(word_idx);
                }
            }
        }

        for (auto& [p, pos_set] : local_pos_updates)
        {
            std::ranges::sort(pos_set);
            pos_set.erase(std::ranges::unique(pos_set).begin(), pos_set.end());
            int32_t cnt = 0;
            if (auto it = pair_counts.find(p); it != pair_counts.end())
            {
                cnt = it->second;
            }

            if (cnt > 0)
            {
                heap.push(MergeJob{p, static_cast<uint64_t>(cnt), std::move(pos_set)});
            }
        }

        ++merges_done;
    }

    vocab_dirty_ = true;
}

void Tokenizer::train(const std::vector<std::string>& texts, const uint32_t vocab_size, const std::optional<std::string>& pattern)
{
    train_from_iterator(texts.begin(), texts.end(), vocab_size, 8192, pattern);
}

std::vector<std::vector<uint8_t>> Tokenizer::build_vocab() const
{
    std::vector<std::vector<uint8_t>> vocab(256);
    for (uint32_t i = 0; i < 256; ++i)
    {
        vocab[i] = {static_cast<uint8_t>(i)};
    }

    std::vector<std::pair<Pair, TokenId>> sorted;
    sorted.reserve(merges_.size());
    for (auto& [p, id] : merges_)
    {
        sorted.emplace_back(p, id);
    }

    std::ranges::sort(sorted, [](const auto& a, const auto& b)
    {
        return a.second < b.second;
    });

    for (auto& [p, merged_id] : sorted)
    {
        if (merged_id >= vocab.size())
        {
            vocab.resize(merged_id + 1);
        }

        auto bytes = vocab.at(p.first);
        const auto& right = vocab.at(p.second);
        bytes.insert(bytes.end(), right.begin(), right.end());
        vocab[merged_id] = std::move(bytes);
    }

    return vocab;
}

std::vector<std::pair<std::vector<uint8_t>, uint32_t>> Tokenizer::get_mergeable_ranks() const
{
    const auto& vocab = cached_vocab();

    std::vector<std::pair<std::vector<uint8_t>, uint32_t>> ranks;
    ranks.reserve(vocab_size());

    for (uint32_t i = 0; i < 256; ++i)
    {
        ranks.emplace_back(vocab[i], i);
    }

    std::vector<std::pair<Pair, TokenId>> sorted;
    sorted.reserve(merges_.size());
    for (auto& [p, id] : merges_)
    {
        sorted.emplace_back(p, id);
    }

    std::ranges::sort(sorted, [](const auto& a, const auto& b)
    {
        return a.second < b.second;
    });

    for (auto &id: sorted | std::views::values)
    {
        ranks.emplace_back(vocab[id], id);
    }

    return ranks;
}

std::vector<TokenId> Tokenizer::encode(const std::string_view text) const
{
    std::vector<TokenId> all_ids;
    all_ids.reserve(text.size() / 6);
    pattern_.for_each_match(text, [&](std::string_view chunk)
    {
        encode_chuck_into(chunk, all_ids);
    });
    return all_ids;
}

void Tokenizer::encode_chuck_into(const std::string_view chunk, std::vector<TokenId>& out) const
{
    const size_t base = out.size();
    out.reserve(base + chunk.size());
    for (const unsigned char c : chunk)
    {
        out.push_back(c);
    }

    const size_t n = out.size() - base;
    if (n < 2) return;

    const auto& emap = cached_encode_map();
    auto ids = std::span<TokenId>(out).subspan(base);
    if (n < 128)
    {
        while (ids.size() >= 2)
        {
            TokenId best_id = UINT32_MAX;
            size_t best_pos = 0;
            bool found = false;

            for (size_t i = 0; i + 1 < ids.size(); ++i)
            {
                const uint64_t key = (static_cast<uint64_t>(ids[i]) << 32) | ids[i + 1];
                if (auto it = emap.find(key); it != emap.end())
                {
                    if (it->second < best_id)
                    {
                        best_id = it->second;
                        best_pos = i;
                        found = true;
                    }
                }
            }
            if (!found)
            {
                break;
            }

            ids[best_pos] = best_id;
            out.erase(out.begin() + base + best_pos + 1);
            ids = std::span<TokenId>(out).subspan(base);
        }
        return;
    }

    const size_t n_large = out.size() - base;
    encode_scratch_.local().nxt.resize(n_large);
    encode_scratch_.local().prv.resize(n_large);
    std::iota(encode_scratch_.local().nxt.begin(), encode_scratch_.local().nxt.end(), 1);
    encode_scratch_.local().prv[0] = n_large;

    for (size_t i = 1; i < n_large; ++i)
    {
        encode_scratch_.local().prv[i] = i - 1;
    }

    encode_scratch_.local().alive.assign(n_large, true);
    auto& heap = encode_scratch_.local().heap;
    while (!heap.empty()) heap.pop();

    for (size_t i = 0; i + 1 < n_large; ++i)
    {
        const uint64_t key = (static_cast<uint64_t>(out[i]) << 32) | out[i + 1];
        if (auto it = emap.find(key); it != emap.end())
        {
            heap.emplace(it->second, i);
        }
    }

    while (!heap.empty())
    {
        auto [rank, pos] = heap.top(); heap.pop();
        if (!encode_scratch_.local().alive[pos])
        {
            continue;
        }
        const size_t r = encode_scratch_.local().nxt[pos];
        if (r == n_large || !encode_scratch_.local().alive[r])
        {
            continue;
        }
        const uint64_t key = (static_cast<uint64_t>(out[base+pos]) << 32) | out[base+r];
        auto it = emap.find(key);
        if (it == emap.end() || it->second != rank)
        {
            continue;
        }

        out[base+pos] = rank;
        encode_scratch_.local().alive[r] = false;
        encode_scratch_.local().nxt[pos] = encode_scratch_.local().nxt[r];
        if (encode_scratch_.local().nxt[r] < n_large)
        {
            encode_scratch_.local().prv[encode_scratch_.local().nxt[r]] = pos;
        }

        if (encode_scratch_.local().nxt[pos] < n_large && encode_scratch_.local().alive[encode_scratch_.local().nxt[pos]])
        {
            const uint64_t key = (static_cast<uint64_t>(out[base+pos]) << 32) | out[base+encode_scratch_.local().nxt[pos]];
            if (auto it = emap.find(key); it != emap.end())
            {
                heap.emplace(it->second, pos);
            }
        }

        if (encode_scratch_.local().prv[pos] < n_large && encode_scratch_.local().alive[encode_scratch_.local().prv[pos]])
        {
            const uint64_t key = (static_cast<uint64_t>(out[base+encode_scratch_.local().prv[pos]]) << 32) | out[base+pos];
            if (auto it = emap.find(key); it != emap.end())
            {
                heap.emplace(it->second, encode_scratch_.local().prv[pos]);
            }
        }
    }

    size_t write_pos = base;
    for (size_t i = 0; i < n_large; ++i)
    {
        if (encode_scratch_.local().alive[i])
        {
            out[write_pos++] = out[base+i];
        }
    }
    out.resize(write_pos);
}

const std::vector<std::vector<uint8_t>>& Tokenizer::cached_vocab() const
{
    if (vocab_dirty_)
    {
        vocab_cache_ = build_vocab();
        vocab_dirty_ = false;
    }
    return vocab_cache_;
}

std::vector<TokenId> Tokenizer::encode_chunk(const std::string_view chunk) const
{
    std::vector<TokenId> ids;
    ids.reserve(chunk.size());
    for (const unsigned char c : chunk)
    {
        ids.push_back(c);
    }

    const size_t n = ids.size();
    if (n < 2)
    {
        return ids;
    }

    const auto& emap = cached_encode_map();
    if (n < 128)
    {
        while (ids.size() >= 2)
        {
            TokenId best_id = UINT32_MAX;
            size_t best_pos = 0;
            bool found = false;

            for (size_t i = 0; i + 1 < ids.size(); ++i)
            {
                const uint64_t key = (static_cast<uint64_t>(ids[i]) << 32) | ids[i + 1];
                if (auto it = emap.find(key); it != emap.end())
                {
                    if (it->second < best_id)
                    {
                        best_id = it->second;
                        best_pos = i;
                        found = true;
                    }
                }
            }

            if (!found)
            {
                break;
            }

            ids[best_pos] = best_id;
            ids.erase(ids.begin() + best_pos + 1);
        }

        return ids;
    }

    std::vector<size_t> nxt(n), prv(n);
    std::iota(nxt.begin(), nxt.end(), 1);
    prv[0] = n;
    for (size_t i = 1; i < n; ++i)
    {
        prv[i] = i - 1;
    }

    std::vector<bool> alive(n,true);

    using Entry = std::pair<TokenId, size_t>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<>> heap;

    for (size_t i = 0; i + 1 < n; ++i)
    {
        const uint64_t key = (static_cast<uint64_t>(ids[i]) << 32) | ids[i + 1];
        if (auto id = emap.find(key); id != emap.end())
        {
            heap.emplace(id->second, i);
        }
    }

    while (!heap.empty())
    {
        auto [rank, pos] = heap.top(); heap.pop();

        if (!alive[pos])
        {
            continue;
        }

        const size_t r = nxt[pos];
        if (r == n || !alive[r])
        {
            continue;
        }

        const uint64_t key = (static_cast<uint64_t>(ids[pos]) << 32) | ids[r];
        auto it = emap.find(key);
        if (it == emap.end() || it->second != rank)
        {
            continue;
        }

        ids[pos] = rank;
        alive[r] = false;
        nxt[pos] = nxt[r];
        if (nxt[r] < n)
        {
            prv[nxt[r]] = pos;
        }

        if (nxt[pos] < n && alive[nxt[pos]])
        {
            const uint64_t key = (static_cast<uint64_t>(ids[pos]) << 32) | ids[nxt[pos]];
            if (auto id = emap.find(key); id != emap.end())
            {
                heap.emplace(id->second, pos);
            }
        }

        if (prv[pos] < n && alive[prv[pos]])
        {
            const uint64_t key = (static_cast<uint64_t>(ids[prv[pos]]) << 32) | ids[pos];
            if (auto id = emap.find(key); id != emap.end())
            {
                heap.emplace(id->second, prv[pos]);
            }
        }
    }

    std::vector<TokenId> out;
    out.reserve(n);
    for (size_t i = 0; i < n; ++i)
    {
        if (alive[i])
        {
            out.push_back(ids[i]);
        }
    }

    return out;
}

std::vector<std::vector<TokenId>> Tokenizer::batch_encode(const std::vector<std::string>& texts) const
{
    std::vector<std::vector<TokenId>> results(texts.size());

    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, texts.size()),
        [&](const tbb::blocked_range<size_t>& r)
        {
            for (size_t i = r.begin(); i != r.end(); ++i)
            {
                results[i] = encode(texts[i]);
            }
        });

    return results;
}

const absl::flat_hash_map<uint64_t, TokenId>& Tokenizer::cached_encode_map() const
{
    if (encode_map_dirty_)
    {
        encode_map_.clear();
        encode_map_.reserve(merges_.size());
        for (const auto& [p, id] : merges_)
        {
            const uint64_t key = (static_cast<uint64_t>(p.first) << 32) | p.second;
            encode_map_[key] = id;
        }
        encode_map_dirty_ = false;
    }
    return encode_map_;
}

std::string Tokenizer::decode(const std::vector<TokenId>& ids) const
{
    const auto& vocab = cached_vocab();

    std::string out;
    for (const TokenId id : ids)
    {
        if (id >= vocab_size() || vocab[id].empty())
        {
            throw std::runtime_error("invalid token id in decode: " + std::to_string(id));
        }

        const auto& bytes = vocab[id];
        out.append(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    }

    for (size_t i = 0; i < out.size();)
    {
        auto c = static_cast<unsigned char>(out[i]);
        int extra = 0;
        if (c < 0x80) { extra = 0; }
        else if ((c & 0xE0) == 0xC0) { extra = 1; }
        else if ((c & 0xF0) == 0xE0) { extra = 2; }
        else if ((c & 0xF8) == 0xF0) { extra = 3; }
        else
        {
            throw std::runtime_error("invalid UTF-8 sequence in decode output");
        }

        for (int j = 1; j <= extra; ++j)
        {
            if (i + j >= out.size() || (static_cast<unsigned char>(out[i + j]) & 0xC0) != 0x80)
            {
                throw std::runtime_error("truncated UTF-8 sequence in decode output");
            }
        }

        i += 1 + extra;
    }

    return out;
}
