package tokencount

import (
	"regexp/syntax"
	"sort"
	"strings"
	"unicode/utf8"
)

// Split patterns of the common tokenizer families, as their tokenizers
// write them. Each has a hand-written matcher for ASCII text, the bulk of
// most prompts, that defers to the regular expression wherever a non-ASCII
// character could change the answer.
var (
	cl100kPattern = `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
	qwen2Pattern  = `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
	o200kPattern  = strings.Join([]string{
		`[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?`,
		`[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?`,
		`\p{N}{1,3}`,
		` ?[^\s\p{L}\p{N}]+[\r\n/]*`,
		`\s*[\r\n]+`,
		`\s+(?!\S)`,
		`\s+`,
	}, "|")
	deepSeekPattern = "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+"
)

// fastMatcher is the end of the piece that starts at pos, or ok false when
// the regular expression must decide.
type fastMatcher func(text string, pos int) (end int, ok bool)

var fastMatchers = map[string]fastMatcher{
	cl100kPattern:   cl100kFast(3),
	qwen2Pattern:    cl100kFast(1),
	o200kPattern:    o200kFast,
	deepSeekPattern: deepSeekFast,
}

func isLetter(c byte) bool { return c|0x20 >= 'a' && c|0x20 <= 'z' }
func isDigit(c byte) bool  { return c >= '0' && c <= '9' }
func isSpace(c byte) bool  { return c == ' ' || (c >= '\t' && c <= '\r') }

// isSymbol: ASCII outside \s, \p{L} and \p{N}, control characters included.
func isSymbol(c byte) bool {
	return c < utf8.RuneSelf && !isLetter(c) && !isDigit(c) && !isSpace(c)
}

// isPunct: ASCII in \p{P} or \p{S}, the 32 printable symbols.
func isPunct(c byte) bool { return c > ' ' && c < 0x7f && !isLetter(c) && !isDigit(c) }

// contraction is the length of (?i:'s|'t|'re|'ve|'m|'ll|'d) at i: 0 when
// there is none, -1 when a non-ASCII character leaves it open (ſ folds to s).
func contraction(text string, i int) int {
	if i+1 >= len(text) || text[i] != '\'' {
		return 0
	}
	c := text[i+1]
	if c >= utf8.RuneSelf {
		return -1
	}
	var second byte
	switch c | 0x20 {
	case 's', 't', 'm', 'd':
		return 2
	case 'r', 'v':
		second = 'e'
	case 'l':
		second = 'l'
	default:
		return 0
	}
	if i+2 >= len(text) {
		return 0
	}
	if d := text[i+2]; d >= utf8.RuneSelf {
		return -1
	} else if d|0x20 == second {
		return 3
	}
	return 0
}

// runEnd is where a run of bytes that satisfy in ends. A non-ASCII byte at
// the end leaves the run open.
func runEnd(text string, start int, in func(byte) bool) (int, bool) {
	end := start
	for end < len(text) && in(text[end]) {
		end++
	}
	return end, end == len(text) || text[end] < utf8.RuneSelf
}

func digitsEnd(text string, pos, limit int) (int, bool) {
	end := pos
	for end < len(text) && end-pos < limit && isDigit(text[end]) {
		end++
	}
	return end, end-pos == limit || end == len(text) || text[end] < utf8.RuneSelf
}

// symbolsEnd matches `[symbol]+[\r\n]*`, with '/' in the tail when slash.
func symbolsEnd(text string, start int, symbol func(byte) bool, slash bool) (int, bool) {
	end, ok := runEnd(text, start, symbol)
	if !ok {
		return 0, false
	}
	for end < len(text) && (text[end] == '\r' || text[end] == '\n' || (slash && text[end] == '/')) {
		end++
	}
	return end, true
}

// spacesEnd matches `\s*[\r\n]+|\s+(?!\S)|\s+` over an ASCII whitespace run.
func spacesEnd(text string, pos int) (int, bool) {
	end, newline := pos, -1
	for end < len(text) && isSpace(text[end]) {
		if text[end] == '\r' || text[end] == '\n' {
			newline = end
		}
		end++
	}
	switch {
	case end < len(text) && text[end] >= utf8.RuneSelf:
		return 0, false
	case newline >= 0:
		return newline + 1, true
	case end < len(text) && end-pos >= 2:
		return end - 1, true
	}
	return end, true
}

// prefixed: a character of [^\r\n\p{L}\p{N}] at pos that an ASCII letter
// follows; open when a non-ASCII character follows.
func prefixed(text string, pos int) (letter, ok bool) {
	if text[pos] == '\r' || text[pos] == '\n' || pos+1 >= len(text) {
		return false, true
	}
	next := text[pos+1]
	return isLetter(next), next < utf8.RuneSelf
}

func cl100kFast(digits int) fastMatcher {
	letters := func(text string, start int) (int, bool) { return runEnd(text, start, isLetter) }
	return func(text string, pos int) (int, bool) {
		c := text[pos]
		switch {
		case c >= utf8.RuneSelf:
			return 0, false
		case isLetter(c):
			return letters(text, pos)
		case isDigit(c):
			return digitsEnd(text, pos, digits)
		}
		if n := contraction(text, pos); n != 0 {
			return pos + n, n > 0
		}
		if letter, ok := prefixed(text, pos); !ok || letter {
			if !ok {
				return 0, false
			}
			return letters(text, pos+1)
		}
		if c == ' ' && pos+1 < len(text) && isSymbol(text[pos+1]) {
			return symbolsEnd(text, pos+1, isSymbol, false)
		}
		if isSymbol(c) {
			return symbolsEnd(text, pos, isSymbol, false)
		}
		return spacesEnd(text, pos)
	}
}

func o200kFast(text string, pos int) (int, bool) {
	c := text[pos]
	switch {
	case c >= utf8.RuneSelf:
		return 0, false
	case isLetter(c):
		return o200kWordEnd(text, pos)
	case isDigit(c):
		return digitsEnd(text, pos, 3)
	}
	if letter, ok := prefixed(text, pos); !ok || letter {
		if !ok {
			return 0, false
		}
		return o200kWordEnd(text, pos+1)
	}
	if c == ' ' && pos+1 < len(text) && isSymbol(text[pos+1]) {
		return symbolsEnd(text, pos+1, isSymbol, true)
	}
	if isSymbol(c) {
		return symbolsEnd(text, pos, isSymbol, true)
	}
	return spacesEnd(text, pos)
}

// o200kWordEnd matches [Upper]*[lower]+ or [Upper]+[lower]*, then an
// optional contraction.
func o200kWordEnd(text string, start int) (int, bool) {
	end, ok := runEnd(text, start, func(c byte) bool { return c-'A' < 26 })
	if !ok {
		return 0, false
	}
	if end, ok = runEnd(text, end, func(c byte) bool { return c-'a' < 26 }); !ok {
		return 0, false
	}
	n := contraction(text, end)
	return end + n, n >= 0
}

func deepSeekFast(text string, pos int) (int, bool) {
	c := text[pos]
	switch {
	case c >= utf8.RuneSelf:
		return 0, false
	case isLetter(c):
		return runEnd(text, pos, isLetter)
	case isPunct(c):
		if pos+1 < len(text) && isLetter(text[pos+1]) {
			end, _ := runEnd(text, pos+1, isLetter)
			return end, true
		}
		return symbolsEnd(text, pos, isPunct, false)
	}
	if letter, ok := prefixed(text, pos); !ok || letter {
		if !ok {
			return 0, false
		}
		return runEnd(text, pos+1, isLetter)
	}
	if c == ' ' && pos+1 < len(text) && isPunct(text[pos+1]) {
		return symbolsEnd(text, pos+1, isPunct, false)
	}
	if isSpace(c) {
		return spacesEnd(text, pos)
	}
	return 0, false
}

// classSplit matches a pattern that is one character class repeated, such
// as \p{N}{1,3}, without the regular expression engine.
type classSplit struct {
	ascii    [2]uint64
	ranges   []rune
	min, max int
}

func newClassSplit(pattern string) *classSplit {
	re, err := syntax.Parse(pattern, syntax.Perl)
	if err != nil {
		return nil
	}
	for re.Op == syntax.OpCapture {
		re = re.Sub[0]
	}
	c := &classSplit{min: 1, max: 1}
	switch re.Op {
	case syntax.OpPlus:
		c.max, re = -1, re.Sub[0]
	case syntax.OpRepeat:
		if re.Flags&syntax.NonGreedy != 0 || re.Min < 1 {
			return nil
		}
		c.min, c.max, re = re.Min, re.Max, re.Sub[0]
	}
	if re.Op != syntax.OpCharClass {
		return nil
	}
	for i := 0; i+1 < len(re.Rune); i += 2 {
		lo, hi := re.Rune[i], re.Rune[i+1]
		for r := lo; r <= hi && r < utf8.RuneSelf; r++ {
			c.ascii[r/64] |= 1 << (r % 64)
		}
		if hi >= utf8.RuneSelf {
			c.ranges = append(c.ranges, max(lo, utf8.RuneSelf), hi)
		}
	}
	return c
}

func (c *classSplit) contains(r rune) bool {
	if r < utf8.RuneSelf {
		return c.ascii[r/64]&(1<<(r%64)) != 0
	}
	i := sort.Search(len(c.ranges)/2, func(i int) bool { return c.ranges[2*i+1] >= r })
	return i < len(c.ranges)/2 && c.ranges[2*i] <= r
}

func (c *classSplit) matchAt(text string, pos int) int {
	end, n := pos, 0
	for end < len(text) && (c.max < 0 || n < c.max) {
		r, size := rune(text[end]), 1
		if r >= utf8.RuneSelf {
			r, size = utf8.DecodeRuneInString(text[end:])
		}
		if !c.contains(r) {
			break
		}
		end += size
		n++
	}
	if n < c.min {
		return pos
	}
	return end
}
