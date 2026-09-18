package cache

import (
	"regexp"
	"strings"
	"unicode"
)

// The lexical tier is the model-free floor of the semantic-cache polarity
// guard (#2691): it rejects near-identical queries that differ only by a
// negation cue or a known antonym swap. Cue-less, word-order-only and
// non-English polarity changes are out of its reach; the NLI tier (#2751)
// covers those.
//
// The tier is unconditional, so a false positive costs recall with no opt-out.
// Cues that are ambiguous in English — "no" abbreviating "number", "on"/"off"
// as prepositions — are gated on their neighbours rather than matched bare.

// tokenDiffLimit bounds polarity checks to near-identical token sequences. A
// cue insertion differs by one token and an antonym swap by two.
const tokenDiffLimit = 2

// negationCues are tokens whose presence on exactly one side flips polarity.
// "n't" contractions are normalized to "not" before tokenization, and "cannot"
// is matched as a whole token. "no" is context-gated; see noIsNegationCue.
var negationCues = map[string]struct{}{
	"not":     {},
	"no":      {},
	"never":   {},
	"without": {},
	"cannot":  {},
}

// antonymFlip is bidirectional. A flip requires opposite tokens in the two
// token differences, so an unpaired antonym does not trigger the guard.
var antonymFlip = buildAntonymFlip([][2]string{
	{"enable", "disable"},
	{"enabled", "disabled"},
	{"on", "off"},
	{"open", "closed"},
	{"open", "close"},
	{"start", "stop"},
	{"add", "remove"},
	{"grant", "revoke"},
	{"increase", "decrease"},
	{"active", "inactive"},
	{"forward", "back"},
	{"forward", "backward"},
})

func buildAntonymFlip(pairs [][2]string) map[string]map[string]struct{} {
	m := make(map[string]map[string]struct{}, len(pairs)*2)
	add := func(a, b string) {
		if m[a] == nil {
			m[a] = make(map[string]struct{})
		}
		m[a][b] = struct{}{}
	}
	for _, p := range pairs {
		add(p[0], p[1])
		add(p[1], p[0])
	}
	return m
}

// Preserve stems that a generic "n't" replacement would corrupt. "ain't" is
// ambiguous, so only its unambiguous negation cue is retained.
var irregularContractions = [][2]string{
	{"can't", "can not"},
	{"won't", "will not"},
	{"shan't", "shall not"},
	{"ain't", "not"},
}

// auxiliaries carry no polarity of their own, and do-support makes them differ
// between a positive and its negation ("works" / "does not work"). Dropping
// them keeps such a pair inside tokenDiffLimit so the cue is actually reached.
// "s" is the residue of an elided "is"/"has" ("it's" tokenizes to it + s).
var auxiliaries = map[string]struct{}{
	"do": {}, "does": {}, "did": {},
	"is": {}, "am": {}, "are": {}, "was": {}, "were": {},
	"be": {}, "been": {}, "being": {},
	"s": {},
}

// toggleVerbs are the verbs that make a following "on"/"off" a state switch
// rather than a preposition.
var toggleVerbs = map[string]struct{}{
	"turn": {}, "turned": {}, "turning": {},
	"switch": {}, "switched": {}, "switching": {},
	"toggle": {}, "toggled": {}, "toggling": {},
	"power": {}, "powered": {}, "powering": {},
	"flip": {}, "flipped": {}, "flipping": {},
	"set": {},
}

// toggleLookbehind is how far before an "on"/"off" token a toggle verb may sit
// and still govern it, covering "turn it on" as well as "turn on".
const toggleLookbehind = 2

// noCompound joins hyphenated "no-" compounds into one token. "no-code" and
// "no-cache" name a thing; splitting them on the hyphen would leave a bare "no"
// that reads as a negation of the rest of the query.
var noCompound = regexp.MustCompile(`\bno-([a-z])`)

// polarityTokens is one query normalized for the guard. seq preserves order so
// ambiguous cues can be gated on their neighbours; set answers membership.
type polarityTokens struct {
	seq []string
	set map[string]struct{}
}

// tokenizeForPolarity normalizes a query into polarity tokens.
func tokenizeForPolarity(s string) polarityTokens {
	s = strings.ToLower(s)
	// Normalize typographic apostrophes before expanding contractions so ASCII
	// and curly-apostrophe contractions take the same negation path.
	s = strings.ReplaceAll(s, "’", "'")
	for _, ic := range irregularContractions {
		s = strings.ReplaceAll(s, ic[0], ic[1])
	}
	s = strings.ReplaceAll(s, "n't", " not")
	// Guarded: the substring test is a plain scan, and almost no query carries a
	// "no-" compound, so the regexp stays off the common path.
	if strings.Contains(s, "no-") {
		s = noCompound.ReplaceAllString(s, "no$1")
	}

	raw := strings.FieldsFunc(s, func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})

	tokens := polarityTokens{
		seq: make([]string, 0, len(raw)),
		set: make(map[string]struct{}, len(raw)),
	}
	for _, tok := range raw {
		// Drop auxiliaries before stemming; "does" and "was" would otherwise be
		// stemmed into non-words.
		if _, aux := auxiliaries[tok]; aux {
			continue
		}
		tok = stemPlural(tok)
		tokens.seq = append(tokens.seq, tok)
		tokens.set[tok] = struct{}{}
	}
	return tokens
}

// stemPlural trims a trailing "s" so a third-person verb and its bare form
// compare equal ("works" / "work"), leaving endings where the "s" belongs to
// the stem ("access", "status", "this", "goes", "uses"). Applied to both sides,
// so at worst it merges two tokens that were already being compared.
func stemPlural(tok string) string {
	if len(tok) <= 3 || !strings.HasSuffix(tok, "s") {
		return tok
	}
	switch tok[len(tok)-2] {
	case 's', 'u', 'i', 'o', 'e':
		return tok
	}
	return tok[:len(tok)-1]
}

// noIsNegationCue reports whether a "no" token in these tokens negates, rather
// than abbreviating "number". "Show invoice no. 123" and "Show invoice no. A123"
// must stay paraphrases of the same query without the abbreviation, so a "no"
// followed by a token carrying a digit is a reference, not a cue.
func (t polarityTokens) noIsNegationCue() bool {
	for i, tok := range t.seq {
		if tok != "no" {
			continue
		}
		if i+1 < len(t.seq) && containsDigit(t.seq[i+1]) {
			continue
		}
		return true
	}
	return false
}

func containsDigit(tok string) bool {
	for _, r := range tok {
		if unicode.IsDigit(r) {
			return true
		}
	}
	return false
}

// togglesState reports whether an "on"/"off" token in these tokens switches a
// state rather than serving as a preposition. "routing based on embeddings"
// must stay a paraphrase of "routing based off embeddings", so the token counts
// only next to a toggle verb or as the final token ("is dark mode on?").
func (t polarityTokens) togglesState(tok string) bool {
	for i, cur := range t.seq {
		if cur != tok {
			continue
		}
		if i == len(t.seq)-1 {
			return true
		}
		for back := 1; back <= toggleLookbehind && i-back >= 0; back++ {
			if _, ok := toggleVerbs[t.seq[i-back]]; ok {
				return true
			}
		}
	}
	return false
}

// hasNegationCue ignores a cue that is ambiguous in the query it came from.
func hasNegationCue(diff []string, from polarityTokens) bool {
	for _, tok := range diff {
		if _, ok := negationCues[tok]; !ok {
			continue
		}
		if tok == "no" && !from.noIsNegationCue() {
			continue
		}
		return true
	}
	return false
}

// diffTokens returns a's distinct tokens that b does not have. Repetition is
// not polarity, so a token repeated in a counts once against tokenDiffLimit.
func diffTokens(a, b polarityTokens) []string {
	var only []string
	seen := make(map[string]struct{}, len(a.seq))
	for _, tok := range a.seq {
		if _, ok := b.set[tok]; ok {
			continue
		}
		if _, dup := seen[tok]; dup {
			continue
		}
		seen[tok] = struct{}{}
		only = append(only, tok)
	}
	return only
}

// polarityMismatch reports whether near-identical queries differ in polarity.
func polarityMismatch(incoming, cached string) bool {
	return polarityMismatchTokens(tokenizeForPolarity(incoming), cached)
}

// polarityMismatchTokens takes the incoming query already tokenized, so a
// caller comparing it against more than one cached entry normalizes it once.
func polarityMismatchTokens(incoming polarityTokens, cached string) bool {
	cachedTokens := tokenizeForPolarity(cached)

	onlyIncoming := diffTokens(incoming, cachedTokens)
	onlyCached := diffTokens(cachedTokens, incoming)

	// Only near-identical, non-identical queries can be polarity variants.
	total := len(onlyIncoming) + len(onlyCached)
	if total == 0 || total > tokenDiffLimit {
		return false
	}

	if hasNegationCue(onlyIncoming, incoming) != hasNegationCue(onlyCached, cachedTokens) {
		return true
	}

	return antonymSwap(onlyIncoming, incoming, onlyCached, cachedTokens)
}

// antonymSwap reports whether the diffs hold opposite tokens of a known pair,
// each used in a sense that actually expresses polarity.
func antonymSwap(onlyIncoming []string, incoming polarityTokens, onlyCached []string, cached polarityTokens) bool {
	cachedDiff := make(map[string]struct{}, len(onlyCached))
	for _, tok := range onlyCached {
		cachedDiff[tok] = struct{}{}
	}
	for _, tok := range onlyIncoming {
		for opp := range antonymFlip[tok] {
			if _, ok := cachedDiff[opp]; !ok {
				continue
			}
			if !antonymSenseHolds(tok, incoming) || !antonymSenseHolds(opp, cached) {
				continue
			}
			return true
		}
	}
	return false
}

// antonymSenseHolds gates the ambiguous entries. Only "on"/"off" are: every
// other pair in antonymFlip carries its polarity on its own.
func antonymSenseHolds(tok string, from polarityTokens) bool {
	switch tok {
	case "on", "off":
		return from.togglesState(tok)
	default:
		return true
	}
}
