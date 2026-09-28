package testcases

import (
	"context"
	"fmt"
	"time"

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

const (
	exactCacheModel    = "e2e-cache-exact"
	exactCacheDecision = "e2e_cache_exact_decision"
)

func init() {
	pkgtestcases.Register("exact-cache-multilingual-negation", pkgtestcases.TestCase{
		Description: "Exact response cache reuses identical requests without serving opposite German or Chinese questions",
		Tags:        []string{"kubernetes", "response-cache", "polarity"},
		Fn:          testExactCacheMultilingualNegation,
	})
}

func testExactCacheMultilingualNegation(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	// The suffix isolates this run from a retained cache while leaving the
	// opposite questions equally similar to their priming questions.
	suffix := fmt.Sprintf(" [cache probe %d]", time.Now().UnixNano())
	queries := []struct {
		language string
		original string
		opposite string
	}{
		{"German", "Ist es sicher, Ibuprofen mit Alkohol zu nehmen?", "Ist es nicht sicher, Ibuprofen mit Alkohol zu nehmen?"},
		{"Chinese", "这个药可以和酒一起吃吗？", "这个药不可以和酒一起吃吗？"},
	}

	exactHits, oppositeMisses := 0, 0
	for _, query := range queries {
		original, opposite := query.original+suffix, query.opposite+suffix
		caseInfo := CacheTestCase{Description: query.language, OriginalQuestion: original}
		for attempt := 0; attempt < cachePolarityPrimeAttempts; attempt++ {
			result := testSingleCacheRequestForModel(ctx, caseInfo, original, localPort, exactCacheModel, opts.Verbose)
			if err := validateExactCacheResult(result); err != nil {
				return fmt.Errorf("%s original request: %w", query.language, err)
			}
		}
		time.Sleep(time.Second)
		repeated := testSingleCacheRequestForModel(ctx, caseInfo, original, localPort, exactCacheModel, opts.Verbose)
		if err := validateExactCacheResult(repeated); err != nil {
			return fmt.Errorf("%s repeated request: %w", query.language, err)
		}
		if !repeated.CacheHit {
			return fmt.Errorf("%s identical request did not hit the exact cache after priming", query.language)
		}
		exactHits++

		result := testSingleCacheRequestForModel(ctx, caseInfo, opposite, localPort, exactCacheModel, opts.Verbose)
		if err := validateExactCacheResult(result); err != nil {
			return fmt.Errorf("%s opposite request: %w", query.language, err)
		}
		if result.CacheHit {
			return fmt.Errorf("%s opposite question received a cached answer: %q", query.language, opposite)
		}
		oppositeMisses++
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"exact_hits":      exactHits,
			"opposite_misses": oppositeMisses,
		})
	}
	return nil
}

func validateExactCacheResult(result CacheResult) error {
	if result.Error != "" {
		return fmt.Errorf("%s", result.Error)
	}
	if result.SelectedRecipe != exactCacheModel || result.SelectedDecision != exactCacheDecision {
		return fmt.Errorf("selected recipe %q and decision %q, expected %q and %q",
			result.SelectedRecipe, result.SelectedDecision, exactCacheModel, exactCacheDecision)
	}
	return nil
}
