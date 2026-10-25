package cache

import (
	"fmt"
	"strings"
	"testing"
	"unicode"
)

// Keep the pre-optimization set algorithm as an independent behavior oracle.
func referencePolarityMismatch(incoming, cached string) bool {
	tokenize := func(s string) map[string]struct{} {
		s = strings.ToLower(s)
		s = strings.ReplaceAll(s, "’", "'")
		for _, contraction := range irregularContractions {
			s = strings.ReplaceAll(s, contraction[0], contraction[1])
		}
		s = strings.ReplaceAll(s, "n't", " not")
		set := make(map[string]struct{})
		for _, token := range strings.FieldsFunc(s, func(r rune) bool {
			return !unicode.IsLetter(r) && !unicode.IsDigit(r)
		}) {
			set[token] = struct{}{}
		}
		return set
	}
	a, b := tokenize(incoming), tokenize(cached)
	difference := func(left, right map[string]struct{}) []string {
		var result []string
		for token := range left {
			if _, ok := right[token]; !ok {
				result = append(result, token)
			}
		}
		return result
	}
	onlyA, onlyB := difference(a, b), difference(b, a)
	if len(onlyA)+len(onlyB) == 0 || len(onlyA)+len(onlyB) > tokenDiffLimit {
		return false
	}
	containsCue := func(tokens []string) bool {
		for _, token := range tokens {
			if _, ok := negationCues[token]; ok {
				return true
			}
		}
		return false
	}
	if containsCue(onlyA) != containsCue(onlyB) {
		return true
	}
	for _, token := range onlyA {
		for _, other := range onlyB {
			if _, ok := antonymFlip[token][other]; ok {
				return true
			}
		}
	}
	return false
}

func TestPreparedPolarityMatchesReference(t *testing.T) {
	corpus := []string{"", "!!!", "can", "can't", "can’t", "will", "won't", "SHAN'T", "ain't", "isn't", "is not", "is not not", "École active", "école inactive", "服務 on", "服務 off"}
	for _, token := range []string{"enable", "disable", "enabled", "disabled", "on", "off", "open", "closed", "close", "start", "stop", "add", "remove", "grant", "revoke", "increase", "decrease", "active", "inactive", "forward", "back", "backward", "not", "no", "never", "without", "cannot"} {
		corpus = append(corpus, "is cache "+token, "CACHE "+strings.ToUpper(token)+" is!!!", token+" cache cache is", "cache is not "+token)
	}
	var long strings.Builder
	for i := 0; i < 80; i++ {
		fmt.Fprintf(&long, " token%d", i)
	}
	corpus = append(corpus, long.String()+" enabled", long.String()+" disabled", long.String()+" not")
	for _, incoming := range corpus {
		for _, cached := range corpus {
			want := referencePolarityMismatch(incoming, cached)
			if got := polarityMismatch(incoming, cached); got != want {
				t.Fatalf("incoming=%q cached=%q: got %t, reference %t", incoming, cached, got, want)
			}
		}
	}
	t.Logf("compared %d ordered query pairs against the original set algorithm", len(corpus)*len(corpus))
}

func TestPreparedPolarityCandidateComparisonDoesNotAllocate(t *testing.T) {
	incoming := tokenizeForPolarity("How do I enable the cache?", nil)
	rejected := tokenizeForPolarity("How do I disable the cache?", nil)
	accepted := tokenizeForPolarity("How can I enable the cache?", nil)
	mismatches := 0
	allocations := testing.AllocsPerRun(100, func() {
		for i := 0; i < 1000; i++ {
			if polarityTokensMismatch(incoming, rejected) {
				mismatches++
			}
			if polarityTokensMismatch(incoming, accepted) {
				t.Fatal("paraphrase rejected")
			}
		}
	})
	if allocations != 0 {
		t.Fatalf("prepared candidate comparisons allocate: %g per scan", allocations)
	}
	if mismatches == 0 {
		t.Fatal("comparison was not exercised")
	}
}
