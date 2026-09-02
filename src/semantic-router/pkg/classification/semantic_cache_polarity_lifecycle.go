package classification

import (
	"context"
	"fmt"

	candle "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// needsSemanticCacheNLIForRuntime reports whether the semantic cache's NLI
// polarity tier (#2751) is configured and therefore needs the NLI model bound.
func (c *Classifier) needsSemanticCacheNLIForRuntime() bool {
	return c != nil && c.Config != nil && c.Config.NeedsLocalNLIForSemanticCache()
}

// initializeSemanticCacheNLI binds the hallucination explainer NLI model for the
// semantic cache polarity guard and injects the verifier into pkg/cache. It is
// registered as a non-best-effort runtime task so a missing or unloadable model
// fails startup instead of silently leaving the guard unwired.
//
// The native binding holds one NLI model (candle.InitNLIModel is Once-guarded),
// so this coexists with HallucinationDetector.InitializeNLI: whichever runs
// first loads the model and the other call is a no-op.
func (c *Classifier) initializeSemanticCacheNLI() error {
	guard := c.Config.SemanticCache.PolarityGuard
	capabilities := CurrentNativeBackendCapabilities()
	if !capabilities.LocalHallucinationNLI {
		return fmt.Errorf(
			"native backend %q does not support local NLI; semantic_cache.polarity_guard mode %q requires the candle backend",
			capabilities.Name, guard.NormalizedMode(),
		)
	}

	nliCfg := c.Config.HallucinationMitigation.NLIModel
	if err := candle.InitNLIModel(nliCfg.ModelID, nliCfg.UseCPU); err != nil {
		return fmt.Errorf("failed to initialize NLI model %q for semantic cache polarity guard: %w", nliCfg.ModelID, err)
	}

	cache.SetPolarityVerifier(c.admittedPolarityVerifier())
	logging.ComponentEvent("classifier", "semantic_cache_nli_initialized", map[string]interface{}{
		"backend":                 "candle",
		"model_ref":               nliCfg.ModelID,
		"mode":                    guard.NormalizedMode(),
		"contradiction_threshold": guard.EffectiveContradictionThreshold(),
	})
	return nil
}

// admittedPolarityVerifier scores the contradiction between a cached query
// (NLI premise) and the incoming query (hypothesis) under the shared
// hallucination-explainer admission gate. Only fields of the result are
// accessed: the cgo binding returns a pointer while the compile-only stub
// returns a value, and this file must build under both.
func (c *Classifier) admittedPolarityVerifier() func(context.Context, string, string) (float32, error) {
	gate := c.admissionRegistry.For(admissionDeploymentHallucinationExplainer)
	return func(ctx context.Context, cachedQuery, incomingQuery string) (float32, error) {
		result, err := admitNLI(ctx, gate, candle.ClassifyNLI, cachedQuery, incomingQuery)
		if err != nil {
			return 0, err
		}
		return result.ContradictProb, nil
	}
}
