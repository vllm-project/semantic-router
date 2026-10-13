package lexical

import (
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification/lexical/internal/deunicode"
)

// tokenize is the bm25 crate's English tokenizer: transliterate to ASCII,
// lowercase, split into Unicode words, drop NLTK English stop words and stem.
func tokenize(text string) []string {
	if text == "" {
		return nil
	}
	words := asciiWords(asciiLower(deunicode.Transliterate(text)))
	tokens := words[:0]
	// A prompt repeats its words; each distinct word is stemmed once.
	stems := make(map[string]string)
	for _, word := range words {
		if _, stop := englishStopWords[word]; stop {
			continue
		}
		stem, ok := stems[word]
		if !ok {
			stem = stemEnglish(word)
			stems[word] = stem
		}
		tokens = append(tokens, stem)
	}
	return tokens
}

func asciiLower(s string) string {
	for i := 0; i < len(s); i++ {
		if 'A' <= s[i] && s[i] <= 'Z' {
			return strings.ToLower(s)
		}
	}
	return s
}

// Word-break classes of ASCII characters (Unicode Standard Annex #29).
type wordBreak uint8

const (
	wbOther wordBreak = iota
	wbCR
	wbLF
	wbNewline
	wbSpace
	wbLetter
	wbNumeric
	wbMidLetter    // :
	wbMidNum       // , ;
	wbMidNumLet    // .
	wbSingleQuote  // '
	wbExtendNumLet // _
)

func asciiWordBreak(c byte) wordBreak {
	switch {
	case 'a' <= c && c <= 'z', 'A' <= c && c <= 'Z':
		return wbLetter
	case '0' <= c && c <= '9':
		return wbNumeric
	}
	switch c {
	case '\r':
		return wbCR
	case '\n':
		return wbLF
	case '\v', '\f':
		return wbNewline
	case ' ':
		return wbSpace
	case ':':
		return wbMidLetter
	case ',', ';':
		return wbMidNum
	case '.':
		return wbMidNumLet
	case '\'':
		return wbSingleQuote
	case '_':
		return wbExtendNumLet
	}
	return wbOther
}

func isMidLetterQ(w wordBreak) bool {
	return w == wbMidLetter || w == wbMidNumLet || w == wbSingleQuote
}

func isMidNumQ(w wordBreak) bool {
	return w == wbMidNum || w == wbMidNumLet || w == wbSingleQuote
}

// asciiWords splits ASCII text at Unicode word boundaries and keeps the
// segments that contain a letter or digit, as unicode-segmentation's
// unicode_words does.
func asciiWords(text string) []string {
	n := len(text)
	if n == 0 {
		return nil
	}
	classes := make([]wordBreak, n)
	for i := 0; i < n; i++ {
		classes[i] = asciiWordBreak(text[i])
	}
	var words []string
	start := 0
	for i := 1; i <= n; i++ {
		if i < n && !wordBoundary(classes, i) {
			continue
		}
		for j := start; j < i; j++ {
			if classes[j] == wbLetter || classes[j] == wbNumeric {
				words = append(words, text[start:i])
				break
			}
		}
		start = i
	}
	return words
}

// wordBoundary reports whether rules WB3 to WB999 break between i-1 and i.
func wordBoundary(c []wordBreak, i int) bool {
	prev, cur := c[i-1], c[i]
	before := wbOther
	if i >= 2 {
		before = c[i-2]
	}
	after := wbOther
	if i+1 < len(c) {
		after = c[i+1]
	}
	switch {
	case prev == wbCR && cur == wbLF: // WB3
		return false
	case prev == wbCR || prev == wbLF || prev == wbNewline, // WB3a
		cur == wbCR || cur == wbLF || cur == wbNewline: // WB3b
		return true
	case prev == wbSpace && cur == wbSpace: // WB3d
		return false
	case prev == wbLetter && cur == wbLetter: // WB5
		return false
	case prev == wbLetter && isMidLetterQ(cur) && after == wbLetter: // WB6
		return false
	case before == wbLetter && isMidLetterQ(prev) && cur == wbLetter: // WB7
		return false
	case (prev == wbNumeric || prev == wbLetter) && (cur == wbNumeric || cur == wbLetter): // WB8-WB10
		return false
	case before == wbNumeric && isMidNumQ(prev) && cur == wbNumeric: // WB11
		return false
	case prev == wbNumeric && isMidNumQ(cur) && after == wbNumeric: // WB12
		return false
	case (prev == wbLetter || prev == wbNumeric || prev == wbExtendNumLet) && cur == wbExtendNumLet: // WB13a
		return false
	case prev == wbExtendNumLet && (cur == wbLetter || cur == wbNumeric): // WB13b
		return false
	}
	return true // WB999
}

// toLowerRust is Rust's str::to_lowercase: full lowercase mappings (İ becomes
// "i̇") and a word-final capital sigma becomes ς.
func toLowerRust(s string) string {
	ascii := true
	for i := 0; i < len(s); i++ {
		if s[i] >= utf8.RuneSelf {
			ascii = false
			break
		}
	}
	if ascii {
		return asciiLower(s)
	}
	var out strings.Builder
	out.Grow(len(s))
	for i, r := range s {
		switch r {
		case 'İ':
			out.WriteString("i\u0307")
		case 'Σ':
			if finalSigma(s, i) {
				out.WriteRune('ς')
			} else {
				out.WriteRune('σ')
			}
		default:
			out.WriteRune(unicode.ToLower(r))
		}
	}
	return out.String()
}

// finalSigma applies the Final_Sigma condition to the Σ at byte offset i.
func finalSigma(s string, i int) bool {
	before := s[:i]
	for before != "" {
		r, size := utf8.DecodeLastRuneInString(before)
		before = before[:len(before)-size]
		if caseIgnorable(r) {
			continue
		}
		if !cased(r) {
			return false
		}
		after := s[i+len("Σ"):]
		for after != "" {
			r, size := utf8.DecodeRuneInString(after)
			after = after[size:]
			if caseIgnorable(r) {
				continue
			}
			return !cased(r)
		}
		return true
	}
	return false
}

func cased(r rune) bool {
	return unicode.In(r, unicode.Lu, unicode.Ll, unicode.Lt, unicode.Other_Lowercase, unicode.Other_Uppercase)
}

func caseIgnorable(r rune) bool {
	switch r {
	case '\'', '.', ':', '·', '\u0387', '\u055f', '\u05f4', '\u2018', '\u2019', '\u2024', '\u2027',
		'\ufe13', '\ufe52', '\ufe55', '\uff07', '\uff0e', '\uff1a':
		return true
	}
	return unicode.In(r, unicode.Mn, unicode.Me, unicode.Cf, unicode.Lm, unicode.Sk)
}

// isAlphanumericRust is Rust's char::is_alphanumeric: the Alphabetic
// property or a numeric general category.
func isAlphanumericRust(r rune) bool {
	if r < utf8.RuneSelf {
		return 'a' <= r && r <= 'z' || 'A' <= r && r <= 'Z' || '0' <= r && r <= '9'
	}
	return unicode.IsLetter(r) || unicode.IsNumber(r) || unicode.Is(unicode.Other_Alphabetic, r)
}
