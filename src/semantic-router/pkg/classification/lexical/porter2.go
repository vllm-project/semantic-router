package lexical

import "slices"

// stemEnglish is the Snowball English (Porter2) stemmer, ported from the
// Snowball-generated code in rust-stemmers 1.2.0, which the bm25 crate used.
// Input is a lowercase ASCII token, as produced by the tokenizer.
func stemEnglish(word string) string {
	if len(word) < 3 {
		return word
	}
	if stem, ok := stemException1(word); ok {
		return stem
	}
	// Snowball English rewrites a word by a few bytes at most; the headroom
	// keeps every replacement in this one buffer.
	s := stemmer{b: append(make([]byte, 0, len(word)+stemHeadroom), word...)}
	s.limit = len(s.b)
	s.prelude()
	s.markRegions()
	s.limitBackward = 0
	s.cursor = s.limit
	s.step1a()
	if !s.exception2() {
		s.cursor = s.limit
		s.step1b()
		s.cursor = s.limit
		s.step1c()
		s.cursor = s.limit
		s.step2()
		s.cursor = s.limit
		s.step3()
		s.cursor = s.limit
		s.step4()
		s.cursor = s.limit
		s.step5()
	}
	if s.yFound {
		for i, c := range s.b {
			if c == 'Y' {
				s.b[i] = 'y'
			}
		}
	}
	if string(s.b) == word {
		return word
	}
	return string(s.b)
}

const stemHeadroom = 8

type stemmer struct {
	b             []byte
	cursor        int
	limit         int
	limitBackward int
	bra, ket      int
	p1, p2        int
	yFound        bool
}

// isVowel is the grouping v (a e i o u y); isVowelWXY adds w, x and Y.
func isVowel(c byte) bool {
	switch c {
	case 'a', 'e', 'i', 'o', 'u', 'y':
		return true
	}
	return false
}

func isVowelWXY(c byte) bool { return isVowel(c) || c == 'w' || c == 'x' || c == 'Y' }

func isValidLI(c byte) bool {
	switch c {
	case 'c', 'd', 'e', 'g', 'h', 'k', 'm', 'n', 'r', 't':
		return true
	}
	return false
}

// replace substitutes b[bra:ket] in place and adjusts the limit and cursor
// the way the Snowball runtime does.
func (s *stemmer) replace(bra, ket int, with string) {
	adjustment := len(with) - (ket - bra)
	tail := len(s.b) - ket
	if adjustment > 0 {
		s.b = slices.Grow(s.b, adjustment)[:len(s.b)+adjustment]
	}
	copy(s.b[bra+len(with):], s.b[ket:ket+tail])
	copy(s.b[bra:], with)
	s.b = s.b[:bra+len(with)+tail]
	s.limit += adjustment
	if s.cursor >= ket {
		s.cursor += adjustment
	} else if s.cursor > bra {
		s.cursor = bra
	}
}

func (s *stemmer) sliceFrom(with string) { s.replace(s.bra, s.ket, with) }

func (s *stemmer) hasSuffix(suffix string) bool {
	start := s.cursor - len(suffix)
	return start >= s.limitBackward && string(s.b[start:s.cursor]) == suffix
}

// findSuffix returns the result of the longest entry that ends at the cursor
// and moves the cursor to its start, or 0 when none matches.
func (s *stemmer) findSuffix(entries []amongEntry) int {
	best := -1
	for i, entry := range entries {
		if s.hasSuffix(entry.text) && (best < 0 || len(entry.text) > len(entries[best].text)) {
			best = i
		}
	}
	if best < 0 {
		return 0
	}
	s.cursor -= len(entries[best].text)
	return entries[best].result
}

type amongEntry struct {
	text   string
	result int
}

var (
	step1aApostrophe = []amongEntry{{"'", 1}, {"'s'", 1}, {"'s", 1}}
	step1aSuffixes   = []amongEntry{{"ied", 2}, {"s", 3}, {"ies", 2}, {"sses", 1}, {"ss", -1}, {"us", -1}}
	step1bEndings    = []amongEntry{{"", 3}, {"bb", 2}, {"dd", 2}, {"ff", 2}, {"gg", 2}, {"bl", 1}, {"mm", 2}, {"nn", 2}, {"pp", 2}, {"rr", 2}, {"at", 1}, {"tt", 2}, {"iz", 1}}
	step1bSuffixes   = []amongEntry{{"ed", 2}, {"eed", 1}, {"ing", 2}, {"edly", 2}, {"eedly", 1}, {"ingly", 2}}
	step2Suffixes    = []amongEntry{
		{"anci", 3},
		{"enci", 2},
		{"ogi", 13},
		{"li", 16},
		{"bli", 12},
		{"abli", 4},
		{"alli", 8},
		{"fulli", 14},
		{"lessli", 15},
		{"ousli", 10},
		{"entli", 5},
		{"aliti", 8},
		{"biliti", 12},
		{"iviti", 11},
		{"tional", 1},
		{"ational", 7},
		{"alism", 8},
		{"ation", 7},
		{"ization", 6},
		{"izer", 6},
		{"ator", 7},
		{"iveness", 11},
		{"fulness", 9},
		{"ousness", 10},
	}
	step2Replacements = map[int]string{
		1: "tion", 2: "ence", 3: "ance", 4: "able", 5: "ent", 6: "ize", 7: "ate", 8: "al", 9: "ful",
		10: "ous", 11: "ive", 12: "ble", 14: "ful", 15: "less",
	}
	step3Suffixes = []amongEntry{
		{"icate", 4},
		{"ative", 6},
		{"alize", 3},
		{"iciti", 4},
		{"ical", 4},
		{"tional", 1},
		{"ational", 2},
		{"ful", 5},
		{"ness", 5},
	}
	step3Replacements = map[int]string{1: "tion", 2: "ate", 3: "al", 4: "ic"}
	step4Suffixes     = []amongEntry{
		{"ic", 1},
		{"ance", 1},
		{"ence", 1},
		{"able", 1},
		{"ible", 1},
		{"ate", 1},
		{"ive", 1},
		{"ize", 1},
		{"iti", 1},
		{"al", 1},
		{"ism", 1},
		{"ion", 2},
		{"er", 1},
		{"ous", 1},
		{"ant", 1},
		{"ent", 1},
		{"ment", 1},
		{"ement", 1},
	}
	exception2Words = map[string]bool{
		"succeed": true, "proceed": true, "exceed": true, "canning": true, "inning": true,
		"earring": true, "herring": true, "outing": true,
	}
	exception1Words = map[string]string{
		"andes": "andes", "atlas": "atlas", "bias": "bias", "cosmos": "cosmos", "dying": "die",
		"early": "earli", "gently": "gentl", "howe": "howe", "idly": "idl", "lying": "lie",
		"news": "news", "only": "onli", "singly": "singl", "skies": "sky", "skis": "ski",
		"sky": "sky", "tying": "tie", "ugly": "ugli",
	}
	regionPrefixes = []string{"gener", "commun", "arsen"}
)

func stemException1(word string) (string, bool) {
	stem, ok := exception1Words[word]
	return stem, ok
}

func (s *stemmer) prelude() {
	if len(s.b) > 0 && s.b[0] == '\'' {
		s.bra, s.ket, s.cursor = 0, 1, 0
		s.sliceFrom("")
	}
	if len(s.b) > 0 && s.b[0] == 'y' {
		s.b[0] = 'Y'
		s.yFound = true
	}
	for i := 0; i+1 < len(s.b); i++ {
		if isVowel(s.b[i]) && s.b[i+1] == 'y' {
			s.b[i+1] = 'Y'
			s.yFound = true
		}
	}
	s.cursor = 0
}

// gopastVowelThenConsonant moves past the next vowel and the consonant after
// it, reporting whether both were found before the limit.
func (s *stemmer) gopastVowelThenConsonant() bool {
	for s.cursor < s.limit && !isVowel(s.b[s.cursor]) {
		s.cursor++
	}
	if s.cursor >= s.limit {
		return false
	}
	s.cursor++
	for s.cursor < s.limit && isVowel(s.b[s.cursor]) {
		s.cursor++
	}
	if s.cursor >= s.limit {
		return false
	}
	s.cursor++
	return true
}

func (s *stemmer) markRegions() {
	s.p1, s.p2 = s.limit, s.limit
	s.cursor = 0
	prefixed := false
	for _, prefix := range regionPrefixes {
		if len(s.b) >= len(prefix) && string(s.b[:len(prefix)]) == prefix {
			s.cursor = len(prefix)
			prefixed = true
			break
		}
	}
	if !prefixed && !s.gopastVowelThenConsonant() {
		s.cursor = 0
		return
	}
	s.p1 = s.cursor
	if s.gopastVowelThenConsonant() {
		s.p2 = s.cursor
	}
	s.cursor = 0
}

// shortv: a short syllable ending at the cursor.
func (s *stemmer) shortv() bool {
	c := s.cursor
	if c-3 >= s.limitBackward && !isVowelWXY(s.b[c-1]) && isVowel(s.b[c-2]) && !isVowel(s.b[c-3]) {
		return true
	}
	return c-2 == s.limitBackward && !isVowel(s.b[c-1]) && isVowel(s.b[c-2])
}

func (s *stemmer) r1() bool { return s.p1 <= s.cursor }
func (s *stemmer) r2() bool { return s.p2 <= s.cursor }

// hasVowelBefore reports whether b[limitBackward:end] contains a vowel.
func (s *stemmer) hasVowelBefore(end int) bool {
	for i := end - 1; i >= s.limitBackward; i-- {
		if isVowel(s.b[i]) {
			return true
		}
	}
	return false
}

func (s *stemmer) step1a() {
	s.ket = s.cursor
	if s.findSuffix(step1aApostrophe) != 0 {
		s.bra = s.cursor
		s.sliceFrom("")
	} else {
		s.cursor = s.limit
	}
	s.ket = s.cursor
	switch s.findSuffix(step1aSuffixes) {
	case 1:
		s.bra = s.cursor
		s.sliceFrom("ss")
	case 2:
		s.bra = s.cursor
		if s.cursor-2 >= s.limitBackward {
			s.sliceFrom("i")
		} else {
			s.sliceFrom("ie")
		}
	case 3:
		s.bra = s.cursor
		// Delete the s if a vowel precedes the letter before it.
		if s.cursor-1 >= s.limitBackward && s.hasVowelBefore(s.cursor-1) {
			s.sliceFrom("")
		}
	}
}

func (s *stemmer) exception2() bool {
	return exception2Words[string(s.b[s.limitBackward:s.limit])]
}

func (s *stemmer) step1b() {
	s.ket = s.cursor
	switch s.findSuffix(step1bSuffixes) {
	case 1:
		s.bra = s.cursor
		if s.r1() {
			s.sliceFrom("ee")
		}
	case 2:
		s.bra = s.cursor
		if !s.hasVowelBefore(s.cursor) {
			return
		}
		s.sliceFrom("")
		end := s.cursor
		switch s.findSuffix(step1bEndings) {
		case 1:
			s.cursor = end
			s.replace(end, end, "e")
		case 2:
			s.cursor = end
			s.ket = end
			s.bra = end - 1
			s.sliceFrom("")
		case 3:
			s.cursor = end
			if end == s.p1 && s.shortv() {
				s.replace(end, end, "e")
			}
		}
	}
}

func (s *stemmer) step1c() {
	s.ket = s.cursor
	if !s.hasSuffix("y") && !s.hasSuffix("Y") {
		return
	}
	s.cursor--
	s.bra = s.cursor
	if s.cursor-1 > s.limitBackward && !isVowel(s.b[s.cursor-1]) {
		s.sliceFrom("i")
	}
}

func (s *stemmer) step2() {
	s.ket = s.cursor
	result := s.findSuffix(step2Suffixes)
	if result == 0 {
		return
	}
	s.bra = s.cursor
	if !s.r1() {
		return
	}
	switch result {
	case 13:
		if s.hasSuffix("l") {
			s.sliceFrom("og")
		}
	case 16:
		if s.cursor-1 >= s.limitBackward && isValidLI(s.b[s.cursor-1]) {
			s.sliceFrom("")
		}
	default:
		s.sliceFrom(step2Replacements[result])
	}
}

func (s *stemmer) step3() {
	s.ket = s.cursor
	result := s.findSuffix(step3Suffixes)
	if result == 0 {
		return
	}
	s.bra = s.cursor
	if !s.r1() {
		return
	}
	switch result {
	case 5:
		s.sliceFrom("")
	case 6:
		if s.r2() {
			s.sliceFrom("")
		}
	default:
		s.sliceFrom(step3Replacements[result])
	}
}

func (s *stemmer) step4() {
	s.ket = s.cursor
	result := s.findSuffix(step4Suffixes)
	if result == 0 {
		return
	}
	s.bra = s.cursor
	if !s.r2() {
		return
	}
	if result == 2 && !s.hasSuffix("s") && !s.hasSuffix("t") {
		return
	}
	s.sliceFrom("")
}

func (s *stemmer) step5() {
	s.ket = s.cursor
	switch {
	case s.hasSuffix("e"):
		s.cursor--
		s.bra = s.cursor
		if s.r2() || (s.r1() && !s.shortv()) {
			s.sliceFrom("")
		}
	case s.hasSuffix("l"):
		s.cursor--
		s.bra = s.cursor
		if s.r2() && s.hasSuffix("l") {
			s.sliceFrom("")
		}
	}
}
