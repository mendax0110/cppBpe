"""
tests/python/test_tokenizer.py

Run after building with CMake:
    cmake --build build --target cppBpe
    cp build/cppBpe*.so tests/python/   # or set PYTHONPATH=build
    pytest tests/python/ -v
"""

import sys
import os
import pytest
import tiktoken
import rustbpe
import regex as re
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../build"))

import cppBpe

def _ref_get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def _ref_merge(ids, pair, idx):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(idx)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out

def _ref_train(text, vocab_size, pattern):
    compiled = re.compile(pattern)
    chunks = re.findall(compiled, text)
    ids = [list(ch.encode("utf-8")) for ch in chunks]
    merges = {}
    for i in range(vocab_size - 256):
        stats = {}
        for chunk_ids in ids:
            _ref_get_stats.__wrapped__ = True
            for pair in zip(chunk_ids, chunk_ids[1:]):
                stats[pair] = stats.get(pair, 0) + 1
        if not stats:
            break
        pair = max(stats, key=stats.get)
        idx = 256 + i
        ids = [_ref_merge(chunk_ids, pair, idx) for chunk_ids in ids]
        merges[pair] = idx
    return merges

def _ref_encode(text, merges, pattern):
    compiled = re.compile(pattern)
    chunks = re.findall(compiled, text)
    result = []
    for chunk in chunks:
        ids = list(chunk.encode("UTF-8"))
        while len(ids) >= 2:
            stats = {pair: 0 for pair in zip(ids, ids[1:])}
            pair = min(stats, key=lambda p: merges.get(p, float("inf")))
            if pair not in merges:
                break
            ids = _ref_merge(ids, pair, merges[pair])
        result.extend(ids)
    return result


class TestConstruction:
    def test_default_tokenizer(self):
        tok = cppBpe.Tokenizer()
        assert tok.vocab_size == 256
        assert tok.get_pattern() == ""

    def test_vocab_size_after_training(self):
        tok = cppBpe.Tokenizer()
        texts = ["ab"] * 10 + ["cd"] * 5
        tok.train_from_iterator(iter(texts), vocab_size=257)
        assert tok.vocab_size == 257

class TestEncode:
    def test_encode_empty(self):
        tok = cppBpe.Tokenizer()
        assert tok.encode("") == []

    def test_encode_basic(self):
        tok = cppBpe.Tokenizer()
        texts = ["ab"] * 10
        tok.train_from_iterator(iter(texts), vocab_size=257, pattern=r"\w+")
        ids = tok.encode("ab")
        assert ids == [256]

    def test_encode_untrained_raw_bytes(self):
        """Without training, encode returns raw byte values (256 base tokens only)."""
        tok = cppBpe.Tokenizer()
        tok.train_from_iterator(iter([]), vocab_size=256)
        ids = tok.encode("hi")
        assert ids == [104, 105]

    def test_encode_roundtrip(self):
        tok = cppBpe.Tokenizer()
        texts = ["hello world "] * 20
        tok.train_from_iterator(iter(texts), vocab_size=300)
        text = "hello world"
        assert tok.decode(tok.encode(text)) == text

    def test_encode_unicode(self):
        tok = cppBpe.Tokenizer()
        tok.train_from_iterator(iter([]), vocab_size=256)
        text = "café"
        # roundtrip must work regardless of merge count
        decoded = tok.decode(tok.encode(text))
        assert decoded == text

class TestBatchEncode:
    def test_batch_encode_matches_single(self):
        tok = cppBpe.Tokenizer()
        texts = ["hi"] * 10 + ["hip"] * 5
        tok.train_from_iterator(iter(texts), vocab_size=257, pattern=r"\w+")

        single = [tok.encode(t) for t in ["hi", "hip", "hi"]]
        batch  = tok.batch_encode(["hi", "hip", "hi"])
        assert batch == single

    def test_batch_encode_empty_list(self):
        tok = cppBpe.Tokenizer()
        assert tok.batch_encode([]) == []

class TestDecode:
    def test_decode_empty(self):
        tok = cppBpe.Tokenizer()
        assert tok.decode([]) == ""

    def test_decode_raw_bytes(self):
        tok = cppBpe.Tokenizer()
        assert tok.decode([104, 105]) == "hi"

    def test_decode_invalid_token_raises(self):
        tok = cppBpe.Tokenizer()
        with pytest.raises(Exception):
            tok.decode([300])  # 300 not in vocab

class TestMergeableRanks:
    def test_base_256_tokens(self):
        tok = cppBpe.Tokenizer()
        ranks = tok.get_mergeable_ranks()
        assert len(ranks) == 256
        # first byte
        b0, r0 = ranks[0]
        assert b0 == bytes([0]) and r0 == 0
        # last byte
        b255, r255 = ranks[255]
        assert b255 == bytes([255]) and r255 == 255

    def test_merge_adds_entry(self):
        tok = cppBpe.Tokenizer()
        texts = ["ab"] * 20
        tok.train_from_iterator(iter(texts), vocab_size=257, pattern=r"\w+")
        ranks = tok.get_mergeable_ranks()
        assert len(ranks) == 257
        # last entry should be b"ab" → 256
        rank_dict = {k: v for k, v in ranks}
        assert rank_dict.get(b"ab") == 256

    def test_tiktoken_export(self):
        """Smoke-test that the output can be used by tiktoken if available."""

        tok = cppBpe.Tokenizer()
        texts = ["hello world "] * 30
        tok.train_from_iterator(iter(texts), vocab_size=300)

        enc = tiktoken.Encoding(
            name="test_enc",
            pat_str=tok.get_pattern(),
            mergeable_ranks={k: v for k, v in tok.get_mergeable_ranks()},
            special_tokens={},
        )
        ids  = enc.encode("hello world")
        text = enc.decode(ids)
        assert text == "hello world"

class TestTraining:
    def test_train_from_list(self):
        tok = cppBpe.Tokenizer()
        tok.train_from_iterator(["hello", "world", "hello"], vocab_size=260)
        assert tok.vocab_size >= 256

    def test_custom_pattern(self):
        tok = cppBpe.Tokenizer()
        texts = ["ab"] * 20
        tok.train_from_iterator(iter(texts), vocab_size=257, pattern=r"\w+")
        assert tok.get_pattern() == r"\w+"
        ids = tok.encode("ab")
        assert ids == [256]

    def test_train_chained_merges(self):
        """'aaa' × 10 with vocab 258 should produce two chained merges."""
        tok = cppBpe.Tokenizer()
        tok.train_from_iterator(["aaa"] * 10, vocab_size=258, pattern=r"\w+")
        assert tok.vocab_size == 258

        # After two merges: 'aaa' should encode to [257]
        ids = tok.encode("aaa")
        assert ids == [257]

    def test_train_most_frequent_merged_first(self):
        """The pair with the highest frequency is merged first."""
        tok = cppBpe.Tokenizer()
        # "ab" × 10, "cd" × 5 -> merge (a,b) first
        texts = ["ab"] * 10 + ["cd"] * 5
        tok.train_from_iterator(iter(texts), vocab_size=257, pattern=r"\w+")
        ranks_dict = {v: k for k, v in tok.get_mergeable_ranks()}
        assert ranks_dict[256] == b"ab"  # first merge must be 'ab'

    def test_empty_iterator_no_merges(self):
        tok = cppBpe.Tokenizer()
        tok.train_from_iterator(iter([]), vocab_size=256)
        assert tok.vocab_size == 256

    def test_buffer_size_param_ignored_gracefully(self):
        """buffer_size is accepted for API parity but does not affect results."""
        tok1 = cppBpe.Tokenizer()
        tok2 = cppBpe.Tokenizer()
        texts = ["hello world"] * 20
        tok1.train_from_iterator(iter(texts), vocab_size=260, buffer_size=4)
        tok2.train_from_iterator(iter(texts), vocab_size=260, buffer_size=65536)
        assert tok1.vocab_size == tok2.vocab_size
        
class TestCorrectnessVsReference:
    """Cross-validate cppBpe against the pure-Python reference."""
    
    UNAMBIGUOUS_TEXT = "ab" * 1000

    def test_merge_order_matches_reference(self):
        """First merge must be the same pair as the Python reference."""
        text = self.UNAMBIGUOUS_TEXT
        vocab_size = 258
        pattern = r"\w+"

        ref_merges = _ref_train(text, vocab_size, pattern)
        first_ref_bytes = bytes(next(
            p for p, idx in ref_merges.items() if idx == 256
        ))

        tok = cppBpe.Tokenizer()
        tok.train_from_iterator([text], vocab_size=vocab_size, pattern=pattern)
        ranks_dict = {v: k for k, v in tok.get_mergeable_ranks()}

        assert ranks_dict[256] == first_ref_bytes

    def test_encode_matches_reference(self):
        """Token IDs must be identical to the Python reference."""
        text = self.UNAMBIGUOUS_TEXT
        vocab_size = 266
        pattern = r"\w+"

        ref_merges = _ref_train(text, vocab_size, pattern)
        ref_ids    = _ref_encode(text[:500], ref_merges, pattern)

        tok = cppBpe.Tokenizer()
        tok.train_from_iterator([text], vocab_size=vocab_size, pattern=pattern)

        assert tok.encode(text[:500]) == ref_ids

    def test_roundtrip_matches_reference(self):
        """decode(encode(text)) must equal text, consistent with the reference."""
        text = self.UNAMBIGUOUS_TEXT[:200]
        vocab_size = 260
        pattern = r"\w+"

        ref_merges = _ref_train(text, vocab_size, pattern)
        ref_ids    = _ref_encode(text, ref_merges, pattern)

        tok = cppBpe.Tokenizer()
        tok.train_from_iterator([text], vocab_size=vocab_size, pattern=pattern)

        assert tok.encode(text) == ref_ids
        assert tok.decode(tok.encode(text)) == text
        
class TestCorrectnessVsRustbpe:
    """Cross-validate cppBpe against rustbpe."""
    
    UNAMBIGOUS_TEXT = "ab" * 1000
    REAL_TEXT = "the quick brown fox jumps over the lazy dog " * 500
    
    def test_encode_matches_rustbpe_unamigous(self):
        """Identical token ID's on strictly unamigous corpus."""
        text = self.UNAMBIGOUS_TEXT
        vocab_size = 266
        pattern = r"\w+"
        
        rust_tok = rustbpe.Tokenizer()
        rust_tok.train_from_iterator([text], vocab_size=vocab_size)
        
        cpp_tok = cppBpe.Tokenizer()
        cpp_tok.train_from_iterator([text], vocab_size=vocab_size, pattern=pattern)
        
        assert cpp_tok.encode(text[:500]) == rust_tok.encode(text[:500])
        
    def test_vocab_overlap_real_text(self):
        """100% vocab overlap on real natrual language text."""
        text = self.REAL_TEXT
        vocab_size = 512
        
        rust_tok = rustbpe.Tokenizer()
        rust_tok.train_from_iterator(text.splitlines(), vocab_size=vocab_size)
        
        cpp_tok = cppBpe.Tokenizer()
        cpp_tok.train_from_iterator(text.splitlines(), vocab_size=vocab_size)
        
        rust_vocab = {bytes(k) for k, _ in rust_tok.get_mergeable_ranks()}
        cpp_vocab = {k for k, _ in cpp_tok.get_mergeable_ranks()}
        
        overlap = len(rust_vocab & cpp_vocab) / len(rust_vocab | cpp_vocab)
        assert overlap == 1.0, f"Vocab overlap was only {overlap:.1f%}"
        
    def test_encode_matches_rustbpe_real_text(self):
        """Identical token ID's on real text at a vocab size where pairs dominate."""
        text = self.REAL_TEXT
        vocab_size = 400
        
        rust_tok = rustbpe.Tokenizer()
        rust_tok.train_from_iterator(text.splitlines(), vocab_size=vocab_size)
        
        cpp_tok = cppBpe.Tokenizer()
        cpp_tok.train_from_iterator(text.splitlines(), vocab_size=vocab_size)
        
        probe = text[:5_000]
        assert cpp_tok.encode(probe) == rust_tok.encode(probe)
        
    def test_tiktoken_export_matches_rustbpe(self):
        """tiktoken encoding from cppBpe must match rustBpe own encoding"""
        text = self.REAL_TEXT
        vocab_size = 400

        rust_tok = rustbpe.Tokenizer()
        rust_tok.train_from_iterator(text.splitlines(), vocab_size=vocab_size)
        
        cpp_tok = cppBpe.Tokenizer()
        cpp_tok.train_from_iterator(text.splitlines(), vocab_size=vocab_size)
        
        enc = tiktoken.Encoding(
            name="cpp_cross_check",
            pat_str=cpp_tok.get_pattern(),
            mergeable_ranks={k: v for k, v in cpp_tok.get_mergeable_ranks()},
            special_tokens={},
        )
        
        probe = text[:5_000]
        assert enc.encode(probe) == rust_tok.encode(probe)
        
        
class TestDeterminism:
    """Training the same data twice must procude identical results."""
    
    def test_train_twice_same_merges(self):
        tok1 = cppBpe.Tokenizer()
        tok2 = cppBpe.Tokenizer()
        texts = ["hello world"] * 50 + ["foo bar baz"] * 30
        tok1.train_from_iterator(iter(texts), vocab_size=280)
        tok2.train_from_iterator(iter(texts), vocab_size=280)
        
        ranks1 = {k: v for k, v in tok1.get_mergeable_ranks()}
        ranks2 = {k: v for k, v in tok1.get_mergeable_ranks()}
        
        assert ranks1 == ranks2
        
    def test_encode_deterministic(self):
        tok = cppBpe.Tokenizer()
        texts = ["hello world"] * 50
        tok.train_from_iterator(iter(texts), vocab_size=270)
        text = "hello world hello"
        assert tok.encode(text) == tok.encode(text)
        
class TestEdgeCases:
    """Edge cases missing from the current suite."""
    
    def test_batch_encode_with_empty_string(self):
        """A batch containing empty strings must not crash."""
        tok = cppBpe.Tokenizer()
        tok.train_from_iterator(["hello"] * 10, vocab_size=260)
        result = tok.batch_encode(["hello", "", "world", ""])
        assert result[1] == []
        assert result[3] == []
        assert result[0] == tok.encode("hello")
        assert result[2] == tok.encode("world")
        
    def test_encode_determinism(self):
        tok = cppBpe.Tokenizer()
        texts = ["hello world"] * 50
        tok.train_from_iterator(iter(texts), vocab_size=270)
        text = "hello world hello"
        assert tok.encode(text) == tok.encode(text)
        
    def test_decode_error_message(self):
        """Invalid token error should mention the bad ID."""
        tok = cppBpe.Tokenizer()
        with pytest.raises(Exception, match="300|invalid|unknown"):
            tok.decode([300])
            
    def test_retrain_resets_vocab(self):
        """Traning a second time on the same tokenizer must replace old merges."""
        tok = cppBpe.Tokenizer()
        tok.train_from_iterator(["ab"] * 20, vocab_size=257, pattern=r"\w+")
        assert tok.vocab_size == 257
        
        tok.train_from_iterator(["cd"] * 20, vocab_size=258, pattern=r"\w+")
        assert tok.vocab_size == 257
        
        ranks_dict = {k: v for k, v in tok.get_mergeable_ranks()}
        assert b"ab" not in ranks_dict
        assert b"cd" in ranks_dict