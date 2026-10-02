#include "itts25/text_processor.hpp"
#include <algorithm>
#include <fstream>
#include <functional>
#include <dlfcn.h>
#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>
#include <cctype>
#include <map>
#include <mutex>
#include <regex>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "kaldifst/csrc/text-normalizer.h"

namespace itts25 {
namespace {

// ---------------------------------------------------------------------------
// UTF-8 helpers
// ---------------------------------------------------------------------------

// Decode one codepoint at byte offset i. Returns number of bytes consumed (>=1)
// and writes the codepoint to cp. Invalid bytes are passed through as 1 byte.
size_t utf8_decode(const std::string& s, size_t i, uint32_t& cp) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    if (c < 0x80) {
        cp = c;
        return 1;
    }
    if ((c >> 5) == 0x6 && i + 1 < s.size()) {
        cp = ((c & 0x1F) << 6) | (static_cast<unsigned char>(s[i + 1]) & 0x3F);
        return 2;
    }
    if ((c >> 4) == 0xE && i + 2 < s.size()) {
        cp = ((c & 0x0F) << 12) | ((static_cast<unsigned char>(s[i + 1]) & 0x3F) << 6) |
             (static_cast<unsigned char>(s[i + 2]) & 0x3F);
        return 3;
    }
    if ((c >> 3) == 0x1E && i + 3 < s.size()) {
        cp = ((c & 0x07) << 18) | ((static_cast<unsigned char>(s[i + 1]) & 0x3F) << 12) |
             ((static_cast<unsigned char>(s[i + 2]) & 0x3F) << 6) |
             (static_cast<unsigned char>(s[i + 3]) & 0x3F);
        return 4;
    }
    cp = c;
    return 1;
}

std::string utf8_encode(uint32_t cp) {
    std::string out;
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    return out;
}

size_t utf8_count(const std::string& s) {
    size_t n = 0;
    for (size_t i = 0; i < s.size();) {
        uint32_t cp = 0;
        i += utf8_decode(s, i, cp);
        ++n;
    }
    return n;
}

// Python str.strip(): trims ASCII + common Unicode whitespace. The reference
// only ever sees ASCII/Chinese, so ASCII whitespace trimming matches.
std::string py_strip(const std::string& s) {
    size_t b = 0, e = s.size();
    auto is_ws = [](unsigned char c) {
        return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
    };
    while (b < e && is_ws(static_cast<unsigned char>(s[b]))) ++b;
    while (e > b && is_ws(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
}

std::string py_rstrip(const std::string& s) {
    size_t e = s.size();
    auto is_ws = [](unsigned char c) {
        return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
    };
    while (e > 0 && is_ws(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(0, e);
}

// ASCII + fullwidth-latin uppercasing (matches Python str.upper() on the
// scripts the reference encounters).
std::string py_upper(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        uint32_t cp = 0;
        size_t n = utf8_decode(s, i, cp);
        if (cp >= 'a' && cp <= 'z') {
            out.push_back(static_cast<char>(cp - 32));
        } else if (cp >= 0xFF41 && cp <= 0xFF5A) {  // fullwidth a-z
            out += utf8_encode(cp - 0x20);
        } else {
            out.append(s, i, n);
        }
        i += n;
    }
    return out;
}

bool is_cjk_basic(uint32_t cp) {  // front.py [一-鿿]
    return cp >= 0x4E00 && cp <= 0x9FFF;
}

// tokenize_by_CJK_char CJK_RANGE_PATTERN ranges.
bool is_cjk_split(uint32_t cp) {
    return (cp >= 0x1100 && cp <= 0x11FF) || (cp >= 0x2E80 && cp <= 0xA4CF) ||
           (cp >= 0xA840 && cp <= 0xD7AF) || (cp >= 0xF900 && cp <= 0xFAFF) ||
           (cp >= 0xFE30 && cp <= 0xFE4F) || (cp >= 0xFF65 && cp <= 0xFFDC) ||
           (cp >= 0x20000 && cp <= 0x2FFFF);
}

bool has_ascii_digit_or_fullwidth(const std::string& s) {
    for (size_t i = 0; i < s.size();) {
        uint32_t cp = 0;
        i += utf8_decode(s, i, cp);
        if (cp >= '0' && cp <= '9') return true;
        if (cp >= 0xFF10 && cp <= 0xFF19) return true;  // fullwidth digits
    }
    return false;
}

// ---------------------------------------------------------------------------
// Punctuation replacement maps (front.py char_rep_map / zh_char_rep_map)
// ---------------------------------------------------------------------------

using ReplPair = std::pair<std::string, std::string>;

const std::vector<ReplPair>& char_rep_map() {
    static const std::vector<ReplPair> m = {
        {"\xEF\xBC\x9A", ","},  // ：
        {"\xEF\xBC\x9B", ","},  // ；
        {";", ","},
        {"\xEF\xBC\x8C", ","},  // ，
        {"\xE3\x80\x82", "."},  // 。
        {"\xEF\xBC\x81", "!"},  // ！
        {"\xEF\xBC\x9F", "?"},  // ？
        {"\n", " "},
        {"\xC2\xB7", "-"},      // ·
        {"\xE3\x80\x81", ","},  // 、
        {"...", "\xE2\x80\xA6"},                  // ... -> …
        {",,,", "\xE2\x80\xA6"},                  // ,,, -> …
        {"\xEF\xBC\x8C\xEF\xBC\x8C\xEF\xBC\x8C", "\xE2\x80\xA6"},  // ，，， -> …
        {"\xE2\x80\xA6\xE2\x80\xA6", "\xE2\x80\xA6"},              // …… -> …
        {"\xE2\x80\x9C", "'"},  // “
        {"\xE2\x80\x9D", "'"},  // ”
        {"\"", "'"},
        {"\xE2\x80\x98", "'"},  // ‘
        {"\xE2\x80\x99", "'"},  // ’
        {"\xEF\xBC\x88", "'"},  // （
        {"\xEF\xBC\x89", "'"},  // ）
        {"(", "'"},
        {")", "'"},
        {"\xE3\x80\x8A", "'"},  // 《
        {"\xE3\x80\x8B", "'"},  // 》
        {"\xE3\x80\x90", "'"},  // 【
        {"\xE3\x80\x91", "'"},  // 】
        {"[", "'"},
        {"]", "'"},
        {"\xE2\x80\x94", "-"},  // —
        {"\xEF\xBD\x9E", "-"},  // ～
        {"~", "-"},
        {"\xE3\x80\x8C", "'"},  // 「
        {"\xE3\x80\x8D", "'"},  // 」
        {":", ","},
    };
    return m;
}

const std::vector<ReplPair>& zh_char_rep_map() {
    static const std::vector<ReplPair> m = [] {
        std::vector<ReplPair> v;
        v.push_back({"$", "."});
        const auto& base = char_rep_map();
        v.insert(v.end(), base.begin(), base.end());
        return v;
    }();
    return m;
}

// Emulate re.compile("|".join(escape(k) for k in keys)).sub(repl): leftmost
// scan, first matching alternative (in map order) wins at each position.
std::string apply_rep_map(const std::string& text, const std::vector<ReplPair>& m) {
    std::string out;
    out.reserve(text.size());
    for (size_t i = 0; i < text.size();) {
        bool matched = false;
        for (const auto& kv : m) {
            const std::string& key = kv.first;
            if (!key.empty() && text.compare(i, key.size(), key) == 0) {
                out += kv.second;
                i += key.size();
                matched = true;
                break;
            }
        }
        if (!matched) {
            out.push_back(text[i]);
            ++i;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Regex-based helpers (contraction / email / pinyin / names)
// ---------------------------------------------------------------------------

std::string apply_contraction(const std::string& text) {
    // (what|where|who|which|how|t?here|it|s?he|that|this)'s  -> \1 is  (icase)
    static const std::regex re(
        R"((what|where|who|which|how|t?here|it|s?he|that|this)'s)",
        std::regex::icase | std::regex::ECMAScript);
    return std::regex_replace(text, re, "$1 is");
}

bool is_email(const std::string& s) {
    static const std::regex re(R"(^[a-zA-Z0-9]+@[a-zA-Z0-9]+\.[a-zA-Z]+$)");
    return std::regex_match(s, re);
}

bool has_ascii_alpha(const std::string& s) {
    for (unsigned char c : s) {
        if (std::isalpha(c)) return true;
    }
    return false;
}

bool has_cjk_basic(const std::string& s) {
    for (size_t i = 0; i < s.size();) {
        uint32_t cp = 0;
        i += utf8_decode(s, i, cp);
        if (is_cjk_basic(cp)) return true;
    }
    return false;
}

// PINYIN_TONE_PATTERN without the (?<![a-z]) lookbehind; the lookbehind is
// enforced by checking the preceding byte is not an ASCII letter.
const std::regex& pinyin_syllable_re() {
    static const std::regex re(
        R"(((?:[bpmfdtnlgkhjqxzcsryw]|[zcs]h)?(?:[aeiou\xC3\xBCv]|[ae]i|u[aio]|ao|ou|i[aue]|[u\xC3\xBCv]e|[uv\xC3\xBC]ang?|uai|[aeiuv]n|[aeio]ng|ia[no]|i[ao]ng)|ng|er)([1-5]))",
        std::regex::icase | std::regex::ECMAScript);
    return re;
}

// Find non-overlapping pinyin-with-tone matches honoring the lookbehind.
std::vector<std::string> find_pinyin(const std::string& text) {
    std::vector<std::string> out;
    auto begin = std::sregex_iterator(text.begin(), text.end(), pinyin_syllable_re());
    auto end = std::sregex_iterator();
    for (auto it = begin; it != end; ++it) {
        const auto& m = *it;
        size_t pos = static_cast<size_t>(m.position(0));
        if (pos > 0) {
            unsigned char prev = static_cast<unsigned char>(text[pos - 1]);
            if (std::isalpha(prev)) continue;  // (?<![a-z]) with icase excludes A-Z too
        }
        out.push_back(m.str(0));
    }
    return out;
}

std::string correct_pinyin(const std::string& pinyin) {
    if (pinyin.empty()) return pinyin;
    char c0 = pinyin[0];
    if (c0 != 'j' && c0 != 'q' && c0 != 'x' && c0 != 'J' && c0 != 'Q' && c0 != 'X') {
        return pinyin;  // not uppercased (matches reference early return)
    }
    // ([jqx])[uü](n|e|an)*(\d) -> \1 v \2 \3   (icase)
    static const std::regex re(R"(([jqx])[u\xC3\xBC]((?:n|e|an)*)(\d))",
                               std::regex::icase | std::regex::ECMAScript);
    std::string fixed = std::regex_replace(pinyin, re, "$1v$2$3");
    return py_upper(fixed);
}

// Replace all non-overlapping occurrences of `needle` in `hay` with `repl`.
std::string replace_all(std::string hay, const std::string& needle, const std::string& repl) {
    if (needle.empty()) return hay;
    size_t pos = 0;
    while ((pos = hay.find(needle, pos)) != std::string::npos) {
        hay.replace(pos, needle.size(), repl);
        pos += repl.size();
    }
    return hay;
}

std::string placeholder(const char* prefix, size_t i) {
    std::string p = "<";
    p += prefix;
    p += static_cast<char>('a' + static_cast<int>(i));
    p += ">";
    return p;
}

// dedup preserving first-seen order
std::vector<std::string> unique_keep_order(const std::vector<std::string>& in) {
    std::vector<std::string> out;
    for (const auto& s : in) {
        if (std::find(out.begin(), out.end(), s) == out.end()) out.push_back(s);
    }
    return out;
}

// NAME_PATTERN: [一-鿿]+([-·—][一-鿿]+){1,2}
std::vector<std::string> find_names(const std::string& text) {
    std::vector<std::string> names;
    // decode to codepoints with byte spans
    std::vector<uint32_t> cps;
    std::vector<size_t> starts;
    std::vector<size_t> lens;
    for (size_t i = 0; i < text.size();) {
        uint32_t cp = 0;
        size_t n = utf8_decode(text, i, cp);
        cps.push_back(cp);
        starts.push_back(i);
        lens.push_back(n);
        i += n;
    }
    auto is_sep = [](uint32_t cp) { return cp == '-' || cp == 0x00B7 || cp == 0x2014; };
    size_t k = 0;
    while (k < cps.size()) {
        if (!is_cjk_basic(cps[k])) {
            ++k;
            continue;
        }
        size_t match_begin = k;
        // first CJK run
        while (k < cps.size() && is_cjk_basic(cps[k])) ++k;
        size_t groups = 0;
        size_t match_end = k;  // exclusive index into cps
        while (groups < 2 && k < cps.size() && is_sep(cps[k]) && k + 1 < cps.size() &&
               is_cjk_basic(cps[k + 1])) {
            ++k;  // consume sep
            while (k < cps.size() && is_cjk_basic(cps[k])) ++k;  // consume CJK run
            ++groups;
            match_end = k;
        }
        if (groups >= 1) {
            size_t bstart = starts[match_begin];
            size_t bend = (match_end < cps.size()) ? starts[match_end]
                                                   : (starts.back() + lens.back());
            names.push_back(text.substr(bstart, bend - bstart));
        }
        // if no group formed, k already advanced past the CJK run; continue
    }
    return names;
}

// ---------------------------------------------------------------------------
// tokenize_by_CJK_char (common.py)
// ---------------------------------------------------------------------------
std::string tokenize_by_cjk_char(const std::string& line_in) {
    std::string line = py_strip(line_in);
    std::vector<std::string> tokens;
    std::string buf;
    auto flush = [&] {
        std::string t = py_strip(buf);
        if (!t.empty()) tokens.push_back(py_upper(t));
        buf.clear();
    };
    for (size_t i = 0; i < line.size();) {
        uint32_t cp = 0;
        size_t n = utf8_decode(line, i, cp);
        if (is_cjk_split(cp)) {
            flush();
            tokens.push_back(py_upper(line.substr(i, n)));  // single CJK char
        } else {
            buf.append(line, i, n);
        }
        i += n;
    }
    flush();
    std::string out;
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i) out.push_back(' ');
        out += tokens[i];
    }
    return out;
}

// ---------------------------------------------------------------------------
// TokenParser (wetext/token_parser.py)
// ---------------------------------------------------------------------------
const char* kEOS = "<EOS>";

using OrderMap = std::map<std::string, std::vector<std::string>>;

const OrderMap& tn_orders() {
    static const OrderMap m = {
        {"date", {"year", "month", "day"}},
        {"fraction", {"denominator", "numerator"}},
        {"measure", {"denominator", "numerator", "value"}},
        {"money", {"value", "currency"}},
        {"time", {"noon", "hour", "minute", "second"}},
    };
    return m;
}

const OrderMap& en_tn_orders() {
    static const OrderMap m = {
        {"date", {"preserve_order", "text", "day", "month", "year"}},
        {"money", {"integer_part", "fractional_part", "quantity", "currency_maj"}},
    };
    return m;
}

struct TokenItem {
    std::string name;
    std::vector<std::string> order;
    std::map<std::string, std::string> members;
    void append(const std::string& k, const std::string& v) {
        order.push_back(k);
        members[k] = v;
    }
    std::string string(const OrderMap& orders) const {
        std::string output = name + " {";
        std::vector<std::string> use_order = order;
        auto it = orders.find(name);
        if (it != orders.end()) {
            auto po = members.find("preserve_order");
            if (po == members.end() || po->second != "true") {
                use_order = it->second;
            }
        }
        for (const auto& key : use_order) {
            auto mit = members.find(key);
            if (mit == members.end()) continue;
            output += " " + key + ": \"" + mit->second + "\"";
        }
        return output + " }";
    }
};

class TokenParser {
public:
    explicit TokenParser(const OrderMap& orders) : orders_(orders) {}

    std::string reorder(const std::string& input) {
        parse(input);
        std::string output;
        for (const auto& tok : tokens_) {
            output += tok.string(orders_) + " ";
        }
        return py_strip(output);
    }

private:
    const OrderMap& orders_;
    std::string text_;
    size_t index_ = 0;
    std::string ch_;  // current "char": a single byte, or kEOS
    std::vector<TokenItem> tokens_;

    void load(const std::string& input) {
        index_ = 0;
        text_ = input;
        ch_ = std::string(1, input[0]);
        tokens_.clear();
    }
    bool read() {
        if (index_ < text_.size() - 1) {
            ++index_;
            ch_ = std::string(1, text_[index_]);
            return true;
        }
        ch_ = kEOS;
        return false;
    }
    bool is_eos() const { return ch_ == kEOS; }
    bool parse_ws() {
        bool not_eos = !is_eos();
        while (not_eos && ch_ == " ") not_eos = read();
        return not_eos;
    }
    bool parse_char(const std::string& exp) {
        if (ch_ == exp) {
            read();
            return true;
        }
        return false;
    }
    void parse_chars(const std::string& exp) {
        for (char x : exp) parse_char(std::string(1, x));
    }
    std::string parse_key() {
        std::string key;
        while (!is_eos() && ch_.size() == 1 &&
               (std::isalpha(static_cast<unsigned char>(ch_[0])) || ch_[0] == '_')) {
            key += ch_;
            read();
        }
        return key;
    }
    std::string parse_value() {
        std::string value;
        bool escape = false;
        while (ch_ != "\"") {
            if (is_eos()) break;  // safety
            value += ch_;
            escape = (ch_ == "\\");
            read();
            if (escape) {
                escape = false;
                value += ch_;
                read();
            }
        }
        return value;
    }
    void parse(const std::string& input) {
        load(input);
        while (parse_ws()) {
            std::string name = parse_key();
            parse_chars(" { ");
            TokenItem token;
            token.name = name;
            while (parse_ws()) {
                if (ch_ == "}") {
                    parse_char("}");
                    break;
                }
                std::string key = parse_key();
                parse_chars(": \"");
                std::string value = parse_value();
                parse_char("\"");
                token.append(key, value);
            }
            tokens_.push_back(std::move(token));
        }
    }
};
std::string sub(const std::string& text,const std::string& pattern,const std::function<std::string(const std::vector<std::string>&)>& replace,bool icase=false) {
    int error;PCRE2_SIZE offset;auto* raw=pcre2_compile(reinterpret_cast<PCRE2_SPTR>(pattern.data()),pattern.size(),PCRE2_UTF|PCRE2_UCP|(icase?PCRE2_CASELESS:0),&error,&offset,nullptr);
    if(!raw)throw std::runtime_error("Invalid frontend regex");std::unique_ptr<pcre2_code,decltype(&pcre2_code_free)> code(raw,pcre2_code_free);
    auto* m=pcre2_match_data_create_from_pattern(raw,nullptr);std::unique_ptr<pcre2_match_data,decltype(&pcre2_match_data_free)> match(m,pcre2_match_data_free);
    size_t pos=0;std::string out;
    while(pos<text.size()) {int n=pcre2_match(raw,reinterpret_cast<PCRE2_SPTR>(text.data()),text.size(),pos,0,m,nullptr);if(n==PCRE2_ERROR_NOMATCH)break;if(n<0)throw std::invalid_argument("Invalid UTF-8 text");auto* v=pcre2_get_ovector_pointer(m);if(v[1]<=v[0])throw std::runtime_error("Empty frontend regex match");std::vector<std::string> groups;for(int i=0;i<n;i++)groups.push_back(v[2*i]==PCRE2_UNSET?"":text.substr(v[2*i],v[2*i+1]-v[2*i]));out+=text.substr(pos,v[0]-pos)+replace(groups);pos=v[1];}
    return out+text.substr(pos);
}
} // namespace
struct TextProcessor::Impl {
    Json config;
    std::unique_ptr<kaldifst::TextNormalizer> zh_tag,zh_verb,en_tag,en_verb;
    void* mecab_lib=nullptr;void* tagger=nullptr;
    const char* (*parse)(void*,const char*,size_t)=nullptr;void (*destroy)(void*)=nullptr;
    explicit Impl(const std::string& dir) {
        std::ifstream in(dir+"/frontend.json");in>>config;
        zh_tag=std::make_unique<kaldifst::TextNormalizer>(dir+"/fsts/zh/tn/tagger.fst");zh_verb=std::make_unique<kaldifst::TextNormalizer>(dir+"/fsts/zh/tn/verbalizer.fst");
        en_tag=std::make_unique<kaldifst::TextNormalizer>(dir+"/fsts/en/tn/tagger.fst");en_verb=std::make_unique<kaldifst::TextNormalizer>(dir+"/fsts/en/tn/verbalizer.fst");
        mecab_lib=dlopen((dir+"/libmecab.2.dylib").c_str(),RTLD_NOW|RTLD_LOCAL);
        if(!mecab_lib)throw std::runtime_error("Missing native MeCab library in frontend resources");
        auto make=reinterpret_cast<void*(*)(int,char**)>(dlsym(mecab_lib,"mecab_new"));parse=reinterpret_cast<decltype(parse)>(dlsym(mecab_lib,"mecab_sparse_tostr2"));destroy=reinterpret_cast<decltype(destroy)>(dlsym(mecab_lib,"mecab_destroy"));
        if(!make||!parse||!destroy)throw std::runtime_error("Invalid MeCab native API");
        std::vector<std::string> args{"itts25","-r",dir+"/unidic/mecabrc","-d",dir+"/unidic","-Owakati"};std::vector<char*> argv;for(auto& a:args)argv.push_back(a.data());
        tagger=make(argv.size(),argv.data());if(!tagger)throw std::runtime_error("Cannot load native UniDic dictionary");
    }
    ~Impl(){if(tagger)destroy(tagger);if(mecab_lib)dlclose(mecab_lib);}
    std::string wetext(std::string text,bool zh) const {
        text=py_strip(text);bool digit=false;sub(text,R"(\d)",[&](const auto& g){digit=true;return g[0];});if(!digit)return text;
        auto tagged=(zh?zh_tag:en_tag)->Normalize(text);if(tagged.empty())throw std::runtime_error("FST text classification failed");TokenParser p(zh?tn_orders():en_tn_orders());auto ordered=p.reorder(py_strip(tagged));return py_strip((zh?zh_verb:en_verb)->Normalize(ordered));
    }
    std::string normalize(std::string text) const {
        std::vector<std::pair<std::string,std::string>> pronunciation;
        text=sub(text,R"(<([^|>\n]+)\|([^>\n]+)>)",[&](const auto& g){int64_t n=pronunciation.size();std::string tag;do{tag=char('a'+n%26)+tag;n=n/26-1;}while(n>=0);auto key="PRONPLACEHOLDER"+tag+"PRONPLACEHOLDER";pronunciation.emplace_back(key,g[0]);return key;});
        bool pinyin=false;sub(text,config.at("pinyin_pattern"),[&](const auto& g){pinyin=true;return g[0];},true);
        bool zh=has_cjk_basic(text)||!has_ascii_alpha(text)||is_email(text)||pinyin;
        text=apply_contraction(text);
        std::vector<std::pair<std::string,Json>> glossary;for(auto it=config.at("glossary").begin();it!=config.at("glossary").end();++it)glossary.emplace_back(it.key(),it.value());
        std::stable_sort(glossary.begin(),glossary.end(),[](const auto& a,const auto& b){return utf8_count(a.first)>utf8_count(b.first);});
        for(const auto& g:glossary){auto repl=g.second.is_string()?g.second.get<std::string>():g.second.value(zh?"zh":"en",g.first);auto escaped=std::regex_replace(g.first,std::regex(R"([.^$|()\[\]{}*+?\\])"),R"(\$&)");text=sub(text,escaped,[&](const auto&){return repl;},true);}
        text=sub(text,R"([A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+)",[](const auto& g){return replace_all(g[0],"-","<H>");});
        std::vector<std::string> pinyins,names;
        if(zh){text=py_rstrip(text);sub(text,config.at("pinyin_pattern"),[&](const auto& g){if(std::find(pinyins.begin(),pinyins.end(),g[0])==pinyins.end())pinyins.push_back(g[0]);return g[0];},true);for(size_t i=0;i<pinyins.size();i++)text=replace_all(text,pinyins[i],placeholder("pinyin_",i));names=unique_keep_order(find_names(text));for(size_t i=0;i<names.size();i++)text=replace_all(text,names[i],placeholder("n_",i));}
        text=wetext(text,zh);
        for(size_t i=0;i<names.size();i++)text=replace_all(text,placeholder("n_",i),names[i]);for(size_t i=0;i<pinyins.size();i++)text=replace_all(text,placeholder("pinyin_",i),correct_pinyin(pinyins[i]));
        text=sub(text,R"(\s*<H>\s*)",[](const auto&){return "-";});text=apply_rep_map(text,zh?zh_char_rep_map():char_rep_map());
        for(const auto& p:pronunciation)text=replace_all(text,p.first,p.second);return text;
    }
    std::string japanese(const std::string& text) const {
        std::string out;size_t pos=0;
        while(pos<text.size()){if(text[pos]==' '){out+=' ';pos++;continue;}size_t end=text.find(' ',pos);if(end==std::string::npos)end=text.size();const char* parsed=parse(tagger,text.data()+pos,end-pos);if(!parsed)throw std::runtime_error("Native MeCab parse failed");out+=py_strip(parsed);pos=end;}return out;
    }
};
TextProcessor::TextProcessor(const std::string& dir):impl_(std::make_unique<Impl>(dir)) {}
TextProcessor::~TextProcessor()=default;
std::string TextProcessor::process(const std::string& raw,const std::string& lang,bool normalization) const {
    auto text=apply_rep_map(raw,char_rep_map());
    if(normalization&&(lang=="zh"||lang=="zhen"||lang=="en"))text=impl_->normalize(text);
    // The installed Python NeMo path has no JA grammar and no available ES grammar.
    if(lang=="zh"||lang=="zhen"||lang=="en"||lang=="ja")text=unicode_transform(text,1);if(lang=="es")text=unicode_transform(text,2);
    text=sub(text,R"(<([^|>\n]+)\|([^>\n]+)>)",[](const auto& g){auto pron=unicode_transform(g[2],2);bool kana=false;for(size_t i=0;i<pron.size();){uint32_t cp;i+=utf8_decode(pron,i,cp);if(cp>=0x3040&&cp<=0x30ff)kana=true;}if(kana)return " "+pron+" ";std::string token=has_cjk_basic(g[1])?"<|SPECIAL_TOKEN_2|>":"<|SPECIAL_TOKEN_1|>";return token+pron+token;});
    if(lang=="ja")text=impl_->japanese(text);return sub(text,R"(<\|([^|]+)\|>)",[](const auto& g){return "<|"+unicode_transform(g[1],2)+"|>";});
}
std::vector<std::string> TextProcessor::split(const std::string& text,const std::string& prefix,const ByteTokenizer& t,uint32_t maximum) const {
    const auto prefix_size=t.encode(prefix).size();size_t cap=std::min(maximum,600u);size_t budget=cap>prefix_size?cap-prefix_size:1;
    auto fits=[&](const std::string& s){return t.encode(s).size()<=budget;};if(fits(text))return {text};
    std::vector<std::string> pieces,chunks;size_t pos=0;
    sub(text,R"(<\|SPECIAL_TOKEN_\d+\|>.*?<\|SPECIAL_TOKEN_\d+\|>)",[&](const auto& g){auto at=text.find(g[0],pos);if(at>pos)pieces.push_back(text.substr(pos,at-pos));pieces.push_back(g[0]);pos=at+g[0].size();return g[0];});if(pos<text.size())pieces.push_back(text.substr(pos));
    for(const auto& piece:pieces){if(piece.rfind("<|SPECIAL_TOKEN_",0)==0){chunks.push_back(piece);continue;}std::vector<std::string> parts;size_t begin=0;
        for(size_t i=0;i<piece.size();){uint32_t cp;size_t n=utf8_decode(piece,i,cp);i+=n;if(cp==0xff0c||cp==0x3002||cp==0xff01||cp==0xff1f||cp==0x3001||cp==0xff1b||cp==0xff1a||std::string(",.!?;:\n").find(cp<128?char(cp):'\0')!=std::string::npos){parts.push_back(piece.substr(begin,i-begin));begin=i;}}if(begin<piece.size())parts.push_back(piece.substr(begin));
        for(const auto& part:parts){if(fits(part)){chunks.push_back(part);continue;}std::string current;for(size_t i=0;i<part.size();){uint32_t cp;size_t n=utf8_decode(part,i,cp);auto ch=part.substr(i,n);i+=n;if(!current.empty()&&!fits(current+ch)){chunks.push_back(current);current=ch;}else current+=ch;}if(!current.empty())chunks.push_back(current);}}
    std::vector<std::string> segments;std::string current;for(const auto& chunk:chunks){if(!current.empty()&&!fits(current+chunk)){segments.push_back(current);current=chunk;}else current+=chunk;}if(!current.empty())segments.push_back(current);return segments;
}
}
