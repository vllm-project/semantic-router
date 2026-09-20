package cache

import "strings"

// semanticCandidateMatchesPolarity applies the same lexical floor to candidates
// fetched from remote stores as the in-memory cache applies during selection.
// A response without its original query cannot establish this contract and is a
// miss. Callers prepare incoming tokens once and keep scanning fetched candidates
// after a rejection; this helper neither changes scores nor runs an NLI model.
func semanticCandidateMatchesPolarity(incoming []string, cachedQuery string) bool {
	if strings.TrimSpace(cachedQuery) == "" {
		return false
	}
	var cachedBuffer [32]string
	return !polarityTokensMismatch(incoming, tokenizeForPolarity(cachedQuery, cachedBuffer[:0]))
}
