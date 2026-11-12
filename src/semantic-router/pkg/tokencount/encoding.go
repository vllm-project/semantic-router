// Package tokencount counts a prompt's tokens inside the Router, with no
// backend involved: byte-level BPE over OpenAI's tiktoken vocabularies, built
// in, or over a model's Hugging Face tokenizer.json.
//
// The count is an estimate. Special tokens in the text are counted as text,
// and a few merge orders of a Hugging Face vocabulary can differ from its
// own tokenizer; callers decide close calls with a margin.
package tokencount

import (
	"context"
	"math"
	"regexp"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"unicode"
	"unicode/utf8"

	"golang.org/x/text/unicode/norm"
)

const (
	// A chunk is the unit of parallel work; smaller texts count in one.
	chunkBytes = 64 << 10
	// A piece longer than this is merged in parts of this size, which
	// bounds the quadratic merge loop on pathological input.
	maxPieceBytes = 512
	// Each chunk caches the counts of the pieces it has seen.
	pieceCacheEntries = 1 << 15
	// The first anchored match looks at this many bytes and grows when a
	// piece ends within matchSlack bytes of the window's end, where a cut
	// could have dropped a suffix such as 're.
	matchWindow = 32
	matchSlack  = 4
)

// Encoding counts the tokens of text with one byte-level BPE vocabulary:
// pre-tokenizer splits, then byte pair merges in rank order.
type Encoding struct {
	name   string
	ranks  map[string]int
	splits []*splitter
	nfc    bool
}

// Name is the encoding's name: a built-in encoding's, or the source of a
// tokenizer.json.
func (e *Encoding) Name() string { return e.name }

// splitter isolates the matches of one pre-tokenizer pattern; text between
// matches is a piece of its own unless dropGaps is set.
type splitter struct {
	re *regexp.Regexp
	// fast or class, when set, answer before re.
	fast     fastMatcher
	class    *classSplit
	dropGaps bool
	// lookahead: the pattern's `\s+(?!\S)` alternative, which RE2 cannot
	// express, was removed. A whitespace run that a non-space follows then
	// leaves its last character to the next piece, as the lookahead does.
	lookahead bool
}

// lookaheadAlternative is the one construct of byte-level BPE patterns that
// Go's RE2 syntax lacks.
const lookaheadAlternative = `\s+(?!\S)|`

func newSplitter(pattern string) (*splitter, error) {
	s := &splitter{fast: fastMatchers[pattern]}
	if i := strings.Index(pattern, lookaheadAlternative); i >= 0 {
		pattern = pattern[:i] + pattern[i+len(lookaheadAlternative):]
		s.lookahead = true
	}
	translated := translatePattern(pattern)
	re, err := regexp.Compile(`^(?:` + translated + `)`)
	if err != nil {
		return nil, err
	}
	s.re = re
	if s.fast == nil && !s.lookahead {
		s.class = newClassSplit(translated)
	}
	return s, nil
}

// The characters of Unicode's White_Space property, which `\s` means to
// tiktoken and Hugging Face tokenizers; RE2's `\s` is ASCII only.
const whiteSpace = `\t\n\v\f\r\x{85}\p{Z}`

// translatePattern rewrites what RE2 reads differently from the Oniguruma
// and fancy-regex syntax of tokenizers: `\s` and `\S` become Unicode's
// whitespace, and possessive quantifiers (`a++`, `a{1,3}+`), which RE2
// lacks, become greedy ones, which split these patterns alike.
func translatePattern(pattern string) string {
	var b strings.Builder
	inClass, quantified := false, false
	for i := 0; i < len(pattern); i++ {
		c := pattern[i]
		if c == '\\' && i+1 < len(pattern) {
			next, end := pattern[i+1], i+2
			if strings.IndexByte("pPx", next) >= 0 && end < len(pattern) && pattern[end] == '{' {
				if k := strings.IndexByte(pattern[end:], '}'); k >= 0 {
					end += k + 1
				}
			}
			switch {
			case next == 's' && inClass:
				b.WriteString(whiteSpace)
			case next == 's':
				b.WriteString(`[` + whiteSpace + `]`)
			case next == 'S' && !inClass:
				b.WriteString(`[^` + whiteSpace + `]`)
			default:
				b.WriteString(pattern[i:end])
			}
			i, quantified = end-1, false
			continue
		}
		switch {
		case inClass:
			inClass = c != ']'
		case c == '[':
			inClass = true
			if strings.HasPrefix(pattern[i+1:], "^]") || strings.HasPrefix(pattern[i+1:], "]") {
				k := strings.IndexByte(pattern[i+1:], ']')
				b.WriteString(pattern[i : i+2+k])
				i += 1 + k
				quantified = false
				continue
			}
		case c == '+' && quantified:
			quantified = false
			continue
		case c == '*' || c == '+' || (c == '?' && (i == 0 || pattern[i-1] != '(')):
			quantified = true
			b.WriteByte(c)
			continue
		case c == '{':
			if k := strings.IndexByte(pattern[i:], '}'); k > 1 && repetition.MatchString(pattern[i+1:i+k]) {
				b.WriteString(pattern[i : i+k+1])
				i += k
				quantified = true
				continue
			}
		}
		quantified = false
		b.WriteByte(c)
	}
	return b.String()
}

var repetition = regexp.MustCompile(`^\d+(,\d*)?$`)

// Count is the number of tokens in text. Large texts are cut at single
// spaces between words, where every supported pattern splits anyway, and
// their parts are counted in parallel.
func (e *Encoding) Count(text string) int {
	tokens, _ := e.CountContext(context.Background(), text)
	return tokens
}

// CountContext is Count that stops early, with ctx's error, once ctx ends.
func (e *Encoding) CountContext(ctx context.Context, text string) (int, error) {
	if text == "" {
		return 0, ctx.Err()
	}
	chunks := splitChunks(text, chunkBytes)
	if len(chunks) == 1 {
		return e.countChunk(text), ctx.Err()
	}
	workers := min(len(chunks), runtime.GOMAXPROCS(0))
	var total atomic.Int64
	var next atomic.Int64
	var wg sync.WaitGroup
	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for ctx.Err() == nil {
				i := int(next.Add(1)) - 1
				if i >= len(chunks) {
					return
				}
				total.Add(int64(e.countChunk(chunks[i])))
			}
		}()
	}
	wg.Wait()
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	return int(total.Load()), nil
}

// splitChunks cuts text into parts of about size bytes, each cut before a
// space that sits between two non-space characters.
func splitChunks(text string, size int) []string {
	if len(text) <= size {
		return []string{text}
	}
	chunks := make([]string, 0, len(text)/size+1)
	start := 0
	for start+size < len(text) {
		cut := -1
		for i := start + size; i < len(text)-1 && i < start+2*size; i++ {
			if text[i] == ' ' && !isSpaceByte(text[i-1]) && !isSpaceByte(text[i+1]) {
				cut = i
				break
			}
		}
		if cut < 0 {
			// No word boundary nearby: cut on a rune boundary instead.
			cut = start + size
			for cut < len(text) && !utf8.RuneStart(text[cut]) {
				cut++
			}
		}
		chunks = append(chunks, text[start:cut])
		start = cut
	}
	return append(chunks, text[start:])
}

func isSpaceByte(b byte) bool {
	return b == ' ' || b == '\t' || b == '\n' || b == '\r' || b == '\v' || b == '\f'
}

func (e *Encoding) countChunk(text string) int {
	if e.nfc {
		text = norm.NFC.String(text)
	}
	cache := make(map[string]int, 1024)
	total := 0
	e.pieces(text, 0, func(piece string) {
		if n, ok := cache[piece]; ok {
			total += n
			return
		}
		n := e.bpeCount(piece)
		if len(cache) < pieceCacheEntries {
			cache[piece] = n
		}
		total += n
	})
	return total
}

// pieces applies the splits from level on and emits the final pieces.
func (e *Encoding) pieces(text string, level int, emit func(string)) {
	if level == len(e.splits) {
		emit(text)
		return
	}
	e.splits[level].each(text, func(piece string) { e.pieces(piece, level+1, emit) })
}

func (s *splitter) each(text string, emit func(string)) {
	gap := -1
	for pos := 0; pos < len(text); {
		end := s.matchAt(text, pos)
		if end <= pos {
			if gap < 0 {
				gap = pos
			}
			_, size := utf8.DecodeRuneInString(text[pos:])
			pos += size
			continue
		}
		if gap >= 0 && !s.dropGaps {
			emit(text[gap:pos])
		}
		gap = -1
		emit(text[pos:end])
		pos = end
	}
	if gap >= 0 && !s.dropGaps {
		emit(text[gap:])
	}
}

// matchAt is the end of the piece that starts at pos, or pos when no
// alternative matches there.
func (s *splitter) matchAt(text string, pos int) int {
	if s.class != nil {
		return s.class.matchAt(text, pos)
	}
	if s.fast != nil {
		if end, ok := s.fast(text, pos); ok {
			return end
		}
	}
	window := matchWindow
	for {
		limit := min(len(text), pos+window)
		for limit > pos+1 && limit < len(text) && !utf8.RuneStart(text[limit]) {
			limit--
		}
		loc := s.re.FindStringIndex(text[pos:limit])
		if loc == nil {
			return pos
		}
		end := pos + loc[1]
		if limit < len(text) && end+matchSlack > limit {
			window *= 4
			continue
		}
		if s.lookahead {
			end = leaveLastSpace(text, pos, end)
		}
		return end
	}
}

// leaveLastSpace applies `\s+(?!\S)`: a run of two or more whitespace
// characters, without line breaks, that a non-space follows ends before its
// last character.
func leaveLastSpace(text string, start, end int) int {
	if end >= len(text) {
		return end
	}
	next, _ := utf8.DecodeRuneInString(text[end:])
	if unicode.IsSpace(next) {
		return end
	}
	runes := 0
	last := end
	for i, r := range text[start:end] {
		if !unicode.IsSpace(r) || r == '\n' || r == '\r' {
			return end
		}
		runes++
		last = start + i
	}
	if runes < 2 {
		return end
	}
	return last
}

// bpeCount is the number of tokens byte pair merging makes of piece, as
// tiktoken merges: always the adjacent pair whose union has the lowest rank.
func (e *Encoding) bpeCount(piece string) int {
	if len(piece) <= 1 {
		return len(piece)
	}
	if _, ok := e.ranks[piece]; ok {
		return 1
	}
	if len(piece) > maxPieceBytes {
		total := 0
		for len(piece) > maxPieceBytes {
			cut := maxPieceBytes
			for cut > 0 && !utf8.RuneStart(piece[cut]) {
				cut--
			}
			if cut == 0 {
				cut = maxPieceBytes
			}
			total += e.bpeCount(piece[:cut])
			piece = piece[cut:]
		}
		return total + e.bpeCount(piece)
	}
	type part struct{ start, rank int }
	var small [24]part
	parts := small[:0]
	if len(piece)+1 > len(small) {
		parts = make([]part, 0, len(piece)+1)
	}
	minRank, minIndex := math.MaxInt, -1
	for i := 0; i < len(piece)-1; i++ {
		rank := e.rank(piece[i : i+2])
		if rank < minRank {
			minRank, minIndex = rank, i
		}
		parts = append(parts, part{i, rank})
	}
	parts = append(parts, part{len(piece) - 1, math.MaxInt}, part{len(piece), math.MaxInt})
	rankAt := func(i int) int {
		if i+3 < len(parts) {
			return e.rank(piece[parts[i].start:parts[i+3].start])
		}
		return math.MaxInt
	}
	for minRank != math.MaxInt {
		i := minIndex
		if i > 0 {
			parts[i-1].rank = rankAt(i - 1)
		}
		parts[i].rank = rankAt(i)
		parts = append(parts[:i+1], parts[i+2:]...)
		minRank, minIndex = math.MaxInt, -1
		for j := 0; j < len(parts)-1; j++ {
			if parts[j].rank < minRank {
				minRank, minIndex = parts[j].rank, j
			}
		}
	}
	return len(parts) - 1
}

func (e *Encoding) rank(token string) int {
	if rank, ok := e.ranks[token]; ok {
		return rank
	}
	return math.MaxInt
}
