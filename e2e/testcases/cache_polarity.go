package testcases

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"time"

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

// The polarity case pins the #2751 contract at the only place E2E can see it:
// with global.stores.response_cache.polarity_guard.mode set to an NLI mode, a
// query that clears the similarity threshold but contradicts the cached one
// must not be served from the cache, while a genuine paraphrase still hits.
func init() {
	pkgtestcases.Register("semantic-cache-polarity", pkgtestcases.TestCase{
		Description: "Semantic cache rejects opposite-meaning queries via the NLI polarity guard",
		Tags:        []string{"kubernetes", "semantic-cache", "polarity"},
		Fn:          testCachePolarity,
	})
}

// cachePolarityCase pairs a primed question with an opposite-meaning variant
// that must miss and a paraphrase that must still hit.
//
// Every question in a case must classify to a decision that carries the
// response_cache plugin and the pair must clear similarity_threshold on the
// profile's embedding model. The shipped cases use everyday wording that the
// envoy-ai-gateway baseline profile classifies as "other" (other_decision,
// which carries the plugin) and were calibrated against a live deployment of
// that profile (mmbert). Domain-flavoured wording such as files, timers, or
// notifications routes to decisions without a cache plugin and would make the
// case vacuous.
type cachePolarityCase struct {
	Description      string `json:"description"`
	OriginalQuestion string `json:"original_question"`
	Contradiction    string `json:"contradiction"`
	Paraphrase       string `json:"paraphrase"`
}

func loadCachePolarityCases(path string) ([]cachePolarityCase, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read polarity cases: %w", err)
	}
	var cases []cachePolarityCase
	if err := json.Unmarshal(data, &cases); err != nil {
		return nil, fmt.Errorf("failed to parse polarity cases: %w", err)
	}
	return cases, nil
}

func testCachePolarity(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing semantic cache NLI polarity guard")
	}

	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	cases, err := loadCachePolarityCases("e2e/testcases/testdata/cache_polarity_cases.json")
	if err != nil {
		return err
	}

	var failures []string
	rejected, served := 0, 0
	for _, tc := range cases {
		outcome := runCachePolarityCase(ctx, tc, localPort, opts.Verbose)
		failures = append(failures, outcome.failures...)
		if outcome.rejected {
			rejected++
		}
		if outcome.served {
			served++
		}
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"cases":                    len(cases),
			"contradictions_rejected":  rejected,
			"paraphrases_served":       served,
			"assertion_failures":       len(failures),
			"similarity_threshold_min": cachePolarityThreshold,
		})
	}

	if len(cases) == 0 {
		return errors.New("polarity test loaded zero cases; nothing was asserted")
	}
	if len(failures) > 0 {
		return fmt.Errorf("semantic cache polarity assertions failed (%d):\n  %s",
			len(failures), strings.Join(failures, "\n  "))
	}
	if opts.Verbose {
		fmt.Printf("[Test] Polarity guard: %d contradictions rejected, %d paraphrases served\n", rejected, served)
	}
	return nil
}

// cachePolarityOutcome records one case: whether the contradiction was rejected
// by the guard, whether the paraphrase was served, and any assertion failures.
type cachePolarityOutcome struct {
	rejected bool
	served   bool
	failures []string
}

// runCachePolarityCase primes the cache with the original question, then
// checks the contradiction and the paraphrase against it.
func runCachePolarityCase(ctx context.Context, tc cachePolarityCase, localPort string, verbose bool) cachePolarityOutcome {
	var out cachePolarityOutcome
	primeCase := CacheTestCase{Description: tc.Description, OriginalQuestion: tc.OriginalQuestion}

	// The in-memory cache is per router replica and the gateway spreads
	// requests across replicas, so a single priming request only warms the
	// replica that happened to serve it. Prime repeatedly to cover them all.
	for i := 0; i < cachePolarityPrimeAttempts; i++ {
		resp, err := sendChatRequest(ctx, tc.OriginalQuestion, localPort, verbose)
		if err != nil {
			out.failures = append(out.failures, fmt.Sprintf("priming %q: %v", tc.OriginalQuestion, err))
			return out
		}
		resp.Body.Close()
	}
	time.Sleep(1 * time.Second) // same settling time as the cache hit-rate case

	contra := probeCachePolarity(ctx, primeCase, tc.Contradiction, localPort, verbose)
	if msg := assertPolarityContradictionRejected(tc, contra); msg != "" {
		out.failures = append(out.failures, msg)
	} else {
		out.rejected = true
	}

	para := probeCachePolarity(ctx, primeCase, tc.Paraphrase, localPort, verbose)
	if msg := assertPolarityParaphraseServed(tc, para); msg != "" {
		out.failures = append(out.failures, msg)
	} else {
		out.served = true
	}
	return out
}

// probeCachePolarity sends one probe. A miss that carries no similarity at all
// means the request reached a replica whose cache holds no candidate — not a
// guard decision — so it is retried once before being judged.
func probeCachePolarity(ctx context.Context, primeCase CacheTestCase, question, localPort string, verbose bool) CacheResult {
	r := testSingleCacheRequest(ctx, primeCase, question, localPort, verbose)
	if r.Error == "" && !r.CacheHit && r.Similarity == 0 {
		if verbose {
			fmt.Printf("[Test] probe %q reached an unprimed replica (no similarity reported); retrying once\n", question)
		}
		r = testSingleCacheRequest(ctx, primeCase, question, localPort, verbose)
	}
	return r
}

// assertPolarityContradictionRejected returns a failure message unless the
// contradiction missed *because of the guard*: a miss whose reported similarity
// is at or above the profile threshold is the only observable proof that the
// NLI tier, not the threshold, produced it.
func assertPolarityContradictionRejected(tc cachePolarityCase, r CacheResult) string {
	switch {
	case r.Error != "":
		return fmt.Sprintf("%q: %s", tc.Contradiction, r.Error)
	case r.CacheHit && r.Similarity >= cachePolarityExactMatch:
		// An identical entry can only come from an earlier run that answered
		// the contradiction itself (a kept cluster); serving it is correct
		// cache behaviour and says nothing about the guard.
		return fmt.Sprintf("%q matched an identical cached entry (similarity=%.4f); the cache was not fresh, rerun against a clean deployment",
			tc.Contradiction, r.Similarity)
	case r.CacheHit:
		return fmt.Sprintf("%q was served the cached answer for %q (similarity=%.4f)",
			tc.Contradiction, tc.OriginalQuestion, r.Similarity)
	case r.Similarity < cachePolarityThreshold:
		return fmt.Sprintf("%q missed below the threshold (similarity=%.4f < %.2f); the polarity guard was not exercised",
			tc.Contradiction, r.Similarity, cachePolarityThreshold)
	default:
		return ""
	}
}

// assertPolarityParaphraseServed returns a failure message unless the
// paraphrase still hit, which pins that the guard preserves legitimate recall.
func assertPolarityParaphraseServed(tc cachePolarityCase, r CacheResult) string {
	switch {
	case r.Error != "":
		return fmt.Sprintf("%q: %s", tc.Paraphrase, r.Error)
	case !r.CacheHit:
		return fmt.Sprintf("paraphrase %q missed (similarity=%.4f); the guard rejected a genuine match",
			tc.Paraphrase, r.Similarity)
	default:
		return ""
	}
}

// cachePolarityPrimeAttempts is how many times the original question is sent
// before probing; see runCachePolarityCase.
const cachePolarityPrimeAttempts = 4

// cachePolarityExactMatch is the similarity of an identical cached query.
const cachePolarityExactMatch = 0.999

// cachePolarityThreshold is the highest similarity_threshold any decision with
// the response_cache plugin uses in the profiles that run this case (the
// envoy-ai-gateway baseline ships 0.8 globally and 0.75 on other_decision). A
// rejected contradiction must report at least this score, otherwise the miss
// came from the threshold and says nothing about the guard.
const cachePolarityThreshold = 0.8
