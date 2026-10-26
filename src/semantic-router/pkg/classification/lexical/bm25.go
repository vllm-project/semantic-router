package lexical

import (
	"math"
	"sort"
)

// BM25 parameters of the bm25 crate. They are variables so every operation
// runs in float32 exactly as the crate computed it.
var (
	bm25K1        float32 = 1.2
	bm25B         float32 = 0.75
	fallbackAvgdl float32 = 256
)

// BM25 scores the text against each keyword of a rule as its own document,
// with Okapi BM25 over the rule's keywords as the corpus.
type BM25 struct {
	rule     Rule
	postings map[string][]bm25Posting
}

// bm25Posting is one keyword containing a token, with the token's IDF times
// its BM25 weight in that keyword.
type bm25Posting struct {
	keyword int
	weight  float32
}

// NewBM25 indexes a rule's keywords. The threshold is a BM25 score.
func NewBM25(rule Rule) (*BM25, error) {
	if err := rule.validate(); err != nil {
		return nil, err
	}
	documents := make([][]string, len(rule.Keywords))
	total := 0
	for i, keyword := range rule.Keywords {
		documents[i] = tokenize(rule.normalizedKeyword(keyword))
		total += len(documents[i])
	}
	avgdl := float32(float64(total) / float64(len(documents)))
	if avgdl <= 0 {
		avgdl = fallbackAvgdl
	}
	frequency := make(map[string]int)
	for _, document := range documents {
		for i, token := range document {
			if firstOccurrence(document, i) {
				frequency[token]++
			}
		}
	}
	corpus := float32(len(documents))
	postings := make(map[string][]bm25Posting, len(frequency))
	for keyword, document := range documents {
		length := float32(len(document))
		for i, token := range document {
			if !firstOccurrence(document, i) {
				continue
			}
			weight := float32(idf(corpus, float32(frequency[token])) * termWeight(occurrences(document, token), length, avgdl))
			postings[token] = append(postings[token], bm25Posting{keyword: keyword, weight: weight})
		}
	}
	return &BM25{rule: rule, postings: postings}, nil
}

func firstOccurrence(tokens []string, i int) bool {
	for _, earlier := range tokens[:i] {
		if earlier == tokens[i] {
			return false
		}
	}
	return true
}

func occurrences(tokens []string, token string) float32 {
	count := 0
	for _, t := range tokens {
		if t == token {
			count++
		}
	}
	return float32(count)
}

// termWeight is tf·(k1+1) / (tf + k1·(1 − b + b·len/avgdl)). The float32
// conversions keep each operation rounded on its own, as in the crate.
func termWeight(frequency, length, avgdl float32) float32 {
	numerator := frequency * (bm25K1 + 1)
	norm := float32(1-bm25B) + float32(bm25B*float32(length/avgdl))
	return numerator / (frequency + float32(bm25K1*norm))
}

// idf is ln(1 + (N − n + 0.5) / (n + 0.5)).
func idf(documents, containing float32) float32 {
	return logf(1 + (documents-containing+0.5)/(containing+0.5))
}

func logf(x float32) float32 { return float32(math.Log(float64(x))) }

// Match scores the text and applies the rule operator. Keywords scoring at
// least the threshold match, best first; equal scores keep declaration order.
func (m *BM25) Match(text *Text) (Match, bool) {
	keywords := m.rule.Keywords
	var scores []float32
	var hit []bool
	for _, token := range text.bm25Tokens(m.rule.CaseSensitive) {
		postings := m.postings[token]
		if len(postings) > 0 && scores == nil {
			scores, hit = make([]float32, len(keywords)), make([]bool, len(keywords))
		}
		for _, posting := range postings {
			scores[posting.keyword] += posting.weight
			hit[posting.keyword] = true
		}
	}
	if scores == nil {
		return m.rule.decide(nil, nil)
	}
	order := make([]int, 0, len(keywords))
	for i := range keywords {
		if hit[i] {
			order = append(order, i)
		}
	}
	sort.SliceStable(order, func(a, b int) bool { return scores[order[a]] > scores[order[b]] })
	var matched []string
	var matchedScores []float32
	for _, i := range order {
		if scores[i] >= m.rule.Threshold {
			matched = append(matched, keywords[i])
			matchedScores = append(matchedScores, scores[i])
		}
	}
	return m.rule.decide(matched, matchedScores)
}
