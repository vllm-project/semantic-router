package extproc

import (
	"hash/fnv"
	"math"
	"strings"
	"unicode"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

// extractContextFeatures produces a fixed-length, deterministic feature
// vector for LinUCB / Linear Thompson Sampling.
//
// We deliberately avoid pulling the request through the embedding subsystem:
//
//   - the embedding pipeline is heavyweight (mmBERT + classifier composers)
//     and routing the bandit through it expands the PR scope to crossing
//     two subsystems for marginal gain at the bandit-state dimensionality
//     (d=64) we want
//   - the bandit only needs *some* deterministic signature of the request
//     that correlates with which model will perform well; cheap surface
//     features (length / question form / language hints / token bigram
//     hashes) are enough to demonstrate the contextual benefit without
//     committing to a particular embedding model
//
// The first 8 dimensions encode hand-rolled signals (query length bucket,
// question marks, code/math hints, prior model phase). The remaining
// dimensions are filled by a hashed-bigram bag-of-tokens projection,
// L2-normalised so quadInv stays bounded.
//
// Feature extraction is pure: same query string -> same vector. This is
// the contract LinUCB needs to be reproducible across replays.
func extractContextFeatures(selCtx *selection.SelectionContext, dim int) []float64 {
	if dim <= 0 {
		return nil
	}
	x := make([]float64, dim)
	if selCtx == nil {
		return x
	}
	query := strings.TrimSpace(selCtx.Query)
	if query == "" {
		return x
	}
	tokens := tokenizeQueryForFeatures(query)
	fillHandRolledSignals(x, dim, query, tokens, selCtx)

	// --- hashed token-bigram projection into the remaining dims ---

	if dim > 8 && len(tokens) > 0 {
		hashedBigramFeatures(tokens, x[8:])
		l2NormaliseInPlace(x[8:])
	}
	return x
}

// tokenizeQueryForFeatures splits on non-letter/digit boundaries and
// lowercases. This is intentionally crude — we just need a stable,
// language-agnostic token stream for hashing.
func tokenizeQueryForFeatures(query string) []string {
	tokens := []string{}
	var current strings.Builder
	flush := func() {
		if current.Len() > 0 {
			tokens = append(tokens, current.String())
			current.Reset()
		}
	}
	for _, r := range strings.ToLower(query) {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			current.WriteRune(r)
			continue
		}
		flush()
	}
	flush()
	return tokens
}

func hashedBigramFeatures(tokens []string, slots []float64) {
	if len(slots) == 0 {
		return
	}
	for i := 0; i+1 < len(tokens); i++ {
		key := tokens[i] + "|" + tokens[i+1]
		h := fnv.New32a()
		_, _ = h.Write([]byte(key))
		idx := int(h.Sum32()) % len(slots)
		if idx < 0 {
			idx += len(slots)
		}
		slots[idx] += 1
	}
	// Also fold unigrams in at half weight, so very short queries with no
	// bigrams still produce a non-trivial signal.
	for _, tok := range tokens {
		h := fnv.New32a()
		_, _ = h.Write([]byte(tok))
		idx := int(h.Sum32()) % len(slots)
		if idx < 0 {
			idx += len(slots)
		}
		slots[idx] += 0.5
	}
}

func l2NormaliseInPlace(values []float64) {
	var norm float64
	for _, v := range values {
		norm += v * v
	}
	if norm <= 0 {
		return
	}
	scale := 1.0 / math.Sqrt(norm)
	for i := range values {
		values[i] *= scale
	}
}

func fillHandRolledSignals(x []float64, dim int, query string, tokens []string, selCtx *selection.SelectionContext) {
	fillBasicSignals(x, dim, query)
	fillContextSignals(x, dim, query, tokens, selCtx)
}

func fillBasicSignals(x []float64, dim int, query string) {
	if dim >= 1 {
		x[0] = clamp01(math.Log1p(float64(len(query))) / math.Log(2048))
	}
	if dim >= 2 && strings.Contains(query, "?") {
		x[1] = 1
	}
	if dim >= 3 && hasCodeHint(query) {
		x[2] = 1
	}
	if dim >= 4 && hasMathHint(query) {
		x[3] = 1
	}
}

func fillContextSignals(x []float64, dim int, query string, tokens []string, selCtx *selection.SelectionContext) {
	if dim >= 5 {
		x[4] = clamp01(math.Log1p(float64(len(tokens))) / math.Log(512))
	}
	if dim >= 6 && selCtx.SessionID != "" {
		x[5] = 1
	}
	if dim >= 7 && selCtx.DecisionName != "" {
		x[6] = 1
	}
	if dim >= 8 {
		x[7] = clamp01(lowercaseRatio(query))
	}
}

func hasCodeHint(query string) bool {
	for _, marker := range codeMarkers {
		if strings.Contains(query, marker) {
			return true
		}
	}
	return false
}

func hasMathHint(query string) bool {
	for _, marker := range mathMarkers {
		if strings.Contains(query, marker) {
			return true
		}
	}
	return false
}

func lowercaseRatio(query string) float64 {
	letters := 0
	lowers := 0
	for _, r := range query {
		if unicode.IsLetter(r) {
			letters++
			if unicode.IsLower(r) {
				lowers++
			}
		}
	}
	if letters == 0 {
		return 0
	}
	return float64(lowers) / float64(letters)
}

var (
	codeMarkers = []string{"```", "()", "{}", "function ", "def ", "import ", "class ", "let ", "var ", "const ", "#include"}
	mathMarkers = []string{"$$", "\\frac", "\\sum", "integral", "derivative", "equation", "matrix"}
)
