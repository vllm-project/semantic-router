package extproc

import (
	"path/filepath"
	"runtime"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestLoadClassifierMappingsSkipsUnusedCoreSignals(t *testing.T) {
	cfg := newCoreSignalMappingGateConfig(t)

	mappings, err := loadClassifierMappings(cfg)
	require.NoError(t, err)
	require.NotNil(t, mappings)
	require.Nil(t, mappings.categoryMapping)
	require.Nil(t, mappings.piiMapping)
	require.Nil(t, mappings.jailbreakMapping)

	components, err := buildRouterComponents(cfg)
	require.NoError(t, err)
	require.NotNil(t, components)
	require.NotNil(t, components.classifier)
	require.NotNil(t, components.classificationSvc)
}

func TestLoadClassifierMappingsRequiresUsedCoreSignalMappings(t *testing.T) {
	tests := []struct {
		name        string
		rule        config.RuleNode
		wantErrPart string
	}{
		{
			name:        "domain signal",
			rule:        config.RuleNode{Type: config.SignalTypeDomain, Name: "billing"},
			wantErrPart: "failed to load category mapping",
		},
		{
			name:        "pii signal",
			rule:        config.RuleNode{Type: config.SignalTypePII, Name: "contains_pii"},
			wantErrPart: "failed to load PII mapping",
		},
		{
			name:        "jailbreak signal",
			rule:        config.RuleNode{Type: config.SignalTypeJailbreak, Name: "detector"},
			wantErrPart: "failed to load jailbreak mapping",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := newCoreSignalMappingGateConfig(t)
			cfg.Decisions = []config.Decision{{
				Name: "guarded-route",
				Rules: config.RuleNode{Operator: "OR", Conditions: []config.RuleNode{
					tt.rule,
				}},
			}}

			_, err := loadClassifierMappings(cfg)
			require.Error(t, err)
			require.Contains(t, err.Error(), tt.wantErrPart)
		})
	}
}

// TestBuildRouterComponentsClosesEarlierResourcesOnLaterFailure asserts
// buildRouterComponents rolls back through its generation when a
// construction step fails: the semantic cache built by an earlier step is
// closed rather than discarded with no reachable handle, so its background
// TTL-cleanup goroutine does not outlive the failed build.
func TestBuildRouterComponentsClosesEarlierResourcesOnLaterFailure(t *testing.T) {
	cfg := &config.RouterConfig{
		SemanticCache: config.SemanticCache{
			Enabled:    true,
			TTLSeconds: 60, // > 0 required to start InMemoryCache's background cleanup goroutine
		},
		InlineModels: config.InlineModels{
			PromptGuard: config.PromptGuardConfig{
				// UseVLLM with no external "guardrail" model configured fails
				// classification.BuildClassifier fast and deterministically,
				// inside createRouterClassifier — the step after the
				// semantic cache has already been built.
				UseVLLM: true,
			},
		},
	}

	baseline := stableGoroutineCount(t)

	components, err := buildRouterComponents(cfg)
	require.Error(t, err)
	require.Nil(t, components)

	requireGoroutinesSettleTo(t, baseline,
		"buildRouterComponents leaked a goroutine when a later construction step failed after the semantic cache was already built")
}

// TestBuildRouterComponentsRepeatedReloadsAreGoroutineStable simulates a
// reload loop — repeated buildRouterComponents + Close cycles — and asserts
// the live goroutine count stays flat rather than growing with each cycle.
// This is the issue's own stability validation ask; run under -race it also
// catches lifecycle data races across repeated builds/closes.
func TestBuildRouterComponentsRepeatedReloadsAreGoroutineStable(t *testing.T) {
	cfg := &config.RouterConfig{
		SemanticCache: config.SemanticCache{
			Enabled:    true,
			TTLSeconds: 60,
		},
	}

	baseline := stableGoroutineCount(t)

	const iterations = 30
	for i := 0; i < iterations; i++ {
		components, err := buildRouterComponents(cfg)
		require.NoError(t, err)
		router := components.buildRouter()
		require.NoError(t, router.Close())
	}

	requireGoroutinesSettleTo(t, baseline,
		"buildRouterComponents+Close leaked goroutines across %d repeated reload cycles", iterations)
}

// stableGoroutineCount returns a goroutine count that has stopped moving, so
// goroutines left winding down by an earlier test are not mistaken for this
// test's baseline. runtime.Gosched is not a barrier — goroutine exit is
// asynchronous — so poll for quiescence rather than guessing a yield count.
func stableGoroutineCount(t *testing.T) int {
	t.Helper()
	var last int
	consecutive := 0
	require.Eventually(t, func() bool {
		runtime.GC()
		current := runtime.NumGoroutine()
		if current == last {
			consecutive++
		} else {
			consecutive = 0
			last = current
		}
		return consecutive >= 3
	}, 10*time.Second, 10*time.Millisecond, "goroutine count never settled")
	return last
}

// requireGoroutinesSettleTo asserts the live goroutine count returns to
// baseline. The resources under test exit deterministically (Close waits for
// them), but unrelated goroutines may still be winding down, so poll instead
// of asserting once.
func requireGoroutinesSettleTo(t *testing.T, baseline int, msg string, args ...interface{}) {
	t.Helper()
	require.Eventuallyf(t, func() bool {
		runtime.GC()
		return runtime.NumGoroutine() <= baseline
	}, 10*time.Second, 10*time.Millisecond, msg, args...)
}

func newCoreSignalMappingGateConfig(t *testing.T) *config.RouterConfig {
	t.Helper()
	missingRoot := filepath.Join(t.TempDir(), "missing-model-assets")
	return &config.RouterConfig{
		InlineModels: config.InlineModels{
			Classifier: config.Classifier{
				CategoryModel: config.CategoryModel{
					ModelID:             "models/mmbert32k-intent-classifier-merged",
					CategoryMappingPath: filepath.Join(missingRoot, "category_mapping.json"),
				},
				PIIModel: config.PIIModel{
					ModelID:        "models/mmbert32k-pii-detector-merged",
					PIIMappingPath: filepath.Join(missingRoot, "pii_type_mapping.json"),
				},
			},
			PromptGuard: config.PromptGuardConfig{
				Enabled:              true,
				ModelID:              "models/mmbert32k-jailbreak-detector-merged",
				JailbreakMappingPath: filepath.Join(missingRoot, "jailbreak_type_mapping.json"),
			},
		},
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{{
				Name:  "default-route",
				Rules: config.RuleNode{Operator: "AND", Conditions: []config.RuleNode{}},
			}},
		},
	}
}
