package lexical

import (
	"sort"
	"strings"
	"unicode/utf8"
)

// ngramResultLimit is the number of corpus matches a search keeps, as the
// binding searched with a limit of 10.
const ngramResultLimit = 10

// Ngram matches text words against a rule's keywords by character n-gram
// similarity (ngrammatic's warp-2 similarity with space padding), which
// tolerates typos.
type Ngram struct {
	rule   Rule
	arity  int
	pad    string
	corpus []ngramEntry
	// grams indexes the corpus: gram -> entries containing it, with counts.
	grams map[string][]gramPosting
}

// ngramEntry is one distinct normalized keyword.
type ngramEntry struct {
	text     string
	length   int   // characters of the padded text
	keywords []int // indexes of the rule keywords with this text
}

type gramPosting struct {
	entry int
	count int
}

// NewNgram indexes a rule's keywords. The threshold is a similarity in [0, 1];
// arities below 2 use 2.
func NewNgram(rule Rule, arity int) (*Ngram, error) {
	if err := rule.validate(); err != nil {
		return nil, err
	}
	if arity < 2 {
		arity = 2
	}
	m := &Ngram{rule: rule, arity: arity, pad: strings.Repeat(" ", arity-1), grams: make(map[string][]gramPosting)}
	byText := make(map[string]int)
	for i, keyword := range rule.Keywords {
		text := rule.normalizedKeyword(keyword)
		if entry, ok := byText[text]; ok {
			m.corpus[entry].keywords = append(m.corpus[entry].keywords, i)
			continue
		}
		entry := len(m.corpus)
		byText[text] = entry
		padded := m.pad + text + m.pad
		m.corpus = append(m.corpus, ngramEntry{text: text, length: utf8.RuneCountInString(padded), keywords: []int{i}})
		for _, gram := range countGrams(padded, arity) {
			m.grams[gram.text] = append(m.grams[gram.text], gramPosting{entry: entry, count: gram.count})
		}
	}
	return m, nil
}

type gramCount struct {
	text  string
	count int
}

// countGrams returns the distinct character n-grams of padded with their
// counts, in first-occurrence order. Like ngrammatic, a text shorter in bytes
// than the arity has no grams.
func countGrams(padded string, arity int) []gramCount {
	if arity > len(padded) {
		return nil
	}
	starts := make([]int, 0, len(padded)+1)
	for i := range padded {
		starts = append(starts, i)
	}
	starts = append(starts, len(padded))
	var grams []gramCount
	var index map[string]int
	for i := 0; i+arity < len(starts); i++ {
		gram := padded[starts[i]:starts[i+arity]]
		if index != nil {
			if at, ok := index[gram]; ok {
				grams[at].count++
				continue
			}
		} else if at := findGram(grams, gram); at >= 0 {
			grams[at].count++
			continue
		}
		grams = append(grams, gramCount{text: gram, count: 1})
		if index == nil && len(grams) > 32 {
			index = make(map[string]int, len(grams)*2)
			for at, g := range grams {
				index[g.text] = at
			}
		} else if index != nil {
			index[gram] = len(grams) - 1
		}
	}
	return grams
}

func findGram(grams []gramCount, gram string) int {
	for i := range grams {
		if grams[i].text == gram {
			return i
		}
	}
	return -1
}

type ngramResult struct {
	entry      int
	similarity float32
}

// search returns up to ngramResultLimit corpus entries whose similarity to
// query is at least the threshold, most similar first; ties keep corpus order.
func (m *Ngram) search(text *Text, query string, same []int) []ngramResult {
	padded := text.ngrams(query, m.arity)
	length := padded.length
	var touched []int
	for _, gram := range padded.grams {
		for _, posting := range m.grams[gram.text] {
			if same[posting.entry] == 0 {
				touched = append(touched, posting.entry)
			}
			same[posting.entry] += min(gram.count, posting.count)
		}
	}
	sort.Ints(touched)
	var results []ngramResult
	for _, entry := range touched {
		shared := same[entry]
		same[entry] = 0
		all := 0
		if length >= m.arity && m.corpus[entry].length >= m.arity {
			all = length + m.corpus[entry].length - 2*m.arity + 2 - shared
		}
		if similarity := ngramSimilarity(shared, all); similarity >= m.rule.Threshold {
			results = append(results, ngramResult{entry: entry, similarity: similarity})
		}
	}
	sort.SliceStable(results, func(i, j int) bool { return results[i].similarity > results[j].similarity })
	if len(results) > ngramResultLimit {
		results = results[:ngramResultLimit]
	}
	return results
}

// ngramSimilarity is ngrammatic's similarity with warp 2:
// (all² − (all − same)²) / all², in float32.
func ngramSimilarity(same, all int) float32 {
	if all == 0 {
		return 0
	}
	allGrams := float32(all)
	diffGrams := allGrams - float32(same)
	squared := allGrams * allGrams
	return (squared - diffGrams*diffGrams) / squared
}

// Match searches every word of the text, then the whole text when keywords
// remain unmatched, and applies the rule operator. Keywords are reported in
// the order they first match.
func (m *Ngram) Match(text *Text) (Match, bool) {
	keywords := m.rule.Keywords
	same := make([]int, len(m.corpus))
	matched := make([]bool, len(keywords))
	var found []string
	var similarities []float32
	collect := func(results []ngramResult) {
		for _, result := range results {
			for _, i := range m.corpus[result.entry].keywords {
				if !matched[i] {
					matched[i] = true
					found = append(found, keywords[i])
					similarities = append(similarities, result.similarity)
				}
			}
		}
	}
	for _, word := range text.ngramWords(m.rule.CaseSensitive) {
		collect(m.search(text, word, same))
	}
	if len(found) < len(keywords) {
		collect(m.search(text, text.input(m.rule.CaseSensitive), same))
	}
	return m.rule.decide(found, similarities)
}
