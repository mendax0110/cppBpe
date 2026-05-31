#include "cppBpe/python_bindings.h"
#include "cppBpe/tokenizer.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <optional>

using namespace cppBpe;
namespace py = pybind11;

namespace
{
    class PyTokenizer
    {
    public:
        PyTokenizer() : tok_()
        {
        }

        void train_from_iterator(const py::object& iterator, const uint32_t vocab_size, const size_t buffer_size = 8192, const std::optional<std::string>& pattern = std::nullopt)
        {
            const py::object iter_obj = py::iter(iterator);
            std::vector<std::string> texts;
            texts.reserve(buffer_size);

            for (py::handle item : iter_obj)
            {
                texts.push_back(item.cast<std::string>());
            }

            py::gil_scoped_release release;
            tok_.train(texts, vocab_size, pattern);
        }

        std::vector<uint32_t> encode(const std::string& text) const
        {
            py::gil_scoped_release release;
            return tok_.encode(text);
        }

        std::string decode(const std::vector<uint32_t>& ids) const
        {
            try
            {
                py::gil_scoped_release release;
                return tok_.decode(ids);
            }
            catch (const std::exception& e)
            {
                throw py::value_error(e.what());
            }
        }

        std::vector<std::vector<uint32_t>> batch_encode(const std::vector<std::string>& texts) const
        {
            py::gil_scoped_release release;
            return tok_.batch_encode(texts);
        }

        uint32_t vocab_size() const noexcept
        {
            return tok_.vocab_size();
        }

        std::string get_pattern() const noexcept
        {
            return tok_.get_patterns();
        }

        std::vector<std::pair<py::bytes, uint32_t>> get_mergeable_ranks() const
        {
            const auto ranks = tok_.get_mergeable_ranks();
            std::vector<std::pair<py::bytes, uint32_t>> py_ranks;
            py_ranks.reserve(ranks.size());

            for (const auto& [bytes_vec, id] : ranks)
            {
                py_ranks.emplace_back(py::bytes(reinterpret_cast<const char*>(bytes_vec.data()), bytes_vec.size()), id);
            }

            return py_ranks;
        }

    private:
        Tokenizer tok_;
    };
}

void cppBpe::register_module(const py::module_& m)
{
    m.doc() = "Fast BPE tokenizer training — C++ port of karpathy/rustbpe";

    py::class_<PyTokenizer>(m, "Tokenizer")
        .def(py::init<>(),
             "Create a new Tokenizer (uses GPT-4 regex pattern by default).")

        .def("train_from_iterator",
             &PyTokenizer::train_from_iterator,
             py::arg("iterator"),
             py::arg("vocab_size"),
             py::arg("buffer_size") = 8192,
             py::arg("pattern")     = py::none(),
             R"doc(
                    Train the tokenizer on an iterator of strings.

                    Parameters
                    ----------
                    iterator : Iterable[str]
                        Any Python iterable that yields strings.
                    vocab_size : int
                        Target vocabulary size (≥ 256).
                    buffer_size : int, optional
                        Kept for API compatibility with rustbpe; unused internally.
                    pattern : str, optional
                        Custom pre-tokenisation regex.  Defaults to the GPT-4 pattern.
             )doc")

        .def("encode",
             &PyTokenizer::encode,
             py::arg("text"),
             "Encode a string into a list of token IDs.")

        .def("decode",
             &PyTokenizer::decode,
             py::arg("ids"),
             "Decode a list of token IDs back to a string.")

        .def("batch_encode",
             &PyTokenizer::batch_encode,
             py::arg("texts"),
             "Encode a list of strings in parallel.  Returns a list of token-ID lists.")

        .def_property_readonly("vocab_size",
             &PyTokenizer::vocab_size,
             "Vocabulary size: 256 (bytes) + number of learned merges.")

        .def("get_pattern",
             &PyTokenizer::get_pattern,
             "Return the regex pattern used for pre-tokenisation.")

        .def("get_mergeable_ranks",
             &PyTokenizer::get_mergeable_ranks,
             R"doc(
                    Return a list of (token_bytes, rank) pairs suitable for passing to tiktoken.

                    Example
                    -------
                    import cppbpe, tiktoken
                    tok = cppbpe.Tokenizer()
                    tok.train_from_iterator(texts, vocab_size=4096)
                    enc = tiktoken.Encoding(
                        name="my_tokenizer",
                        pat_str=tok.get_pattern(),
                        mergeable_ranks={k: v for k, v in tok.get_mergeable_ranks()},
                        special_tokens={},
                    )
             )doc");
}

PYBIND11_MODULE(cppBpe, m)
{
    cppBpe::register_module(m);
}

