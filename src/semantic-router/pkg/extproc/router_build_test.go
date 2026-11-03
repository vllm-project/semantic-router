package extproc

import (
	"errors"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func stickyEnabledDecision(t *testing.T, name string) config.Decision {
	t.Helper()
	return config.Decision{
		Name: name,
		Plugins: []config.DecisionPlugin{
			mustToolSelectionDecisionPlugin(t, &config.ToolSelectionPluginConfig{
				Enabled: true,
				Mode:    config.ToolSelectionModeAdd,
				Sticky:  &config.StickyToolSelectionConfig{Enabled: true},
			}),
		},
	}
}

func stickyDisabledDecision(t *testing.T, name string) config.Decision {
	t.Helper()
	return config.Decision{
		Name: name,
		Plugins: []config.DecisionPlugin{
			mustToolSelectionDecisionPlugin(t, &config.ToolSelectionPluginConfig{
				Enabled: true,
				Mode:    config.ToolSelectionModeAdd,
			}),
		},
	}
}

func TestValidateStickyToolSelectionSecret_NoStickyDecisions_OK(t *testing.T) {
	t.Setenv("USER_SCOPE_NAMESPACE_SECRET", "")
	cfg := &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{stickyDisabledDecision(t, "d1")},
		},
	}
	if err := validateStickyToolSelectionSecret(cfg); err != nil {
		t.Fatalf("no decision enables sticky, expected no error, got: %v", err)
	}
}

func TestValidateStickyToolSelectionSecret_StickyEnabledSecretMissing_Err(t *testing.T) {
	t.Setenv("USER_SCOPE_NAMESPACE_SECRET", "")
	cfg := &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{
				stickyDisabledDecision(t, "d1"),
				stickyEnabledDecision(t, "d2"),
			},
		},
	}
	err := validateStickyToolSelectionSecret(cfg)
	if err == nil {
		t.Fatal("expected an error: sticky enabled with no USER_SCOPE_NAMESPACE_SECRET configured")
	}
	if !strings.Contains(err.Error(), "USER_SCOPE_NAMESPACE_SECRET") ||
		!strings.Contains(err.Error(), "sticky") {
		t.Fatalf("error = %q, want an actionable message naming USER_SCOPE_NAMESPACE_SECRET and sticky", err.Error())
	}
}

func TestValidateStickyToolSelectionSecret_StickyEnabledSecretConfigured_OK(t *testing.T) {
	t.Setenv("USER_SCOPE_NAMESPACE_SECRET", "test-secret")
	cfg := &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{stickyEnabledDecision(t, "d1")},
		},
	}
	if err := validateStickyToolSelectionSecret(cfg); err != nil {
		t.Fatalf("secret is configured, expected no error, got: %v", err)
	}
}

func TestValidateStickyToolSelectionSecret_NilConfig_OK(t *testing.T) {
	t.Setenv("USER_SCOPE_NAMESPACE_SECRET", "")
	if err := validateStickyToolSelectionSecret(nil); err != nil {
		t.Fatalf("nil config, expected no error, got: %v", err)
	}
}

// Router construction must reject a sticky decision the local runtime cannot
// serve, even when config admission was bypassed (for example a Kubernetes
// reconciler handing over an already-parsed configuration) and the secret is
// configured, so only the supported-runtime gate can fail here.
func TestBuildOpenAIRouterFromConfig_RejectsUnsupportedStickyRuntime(t *testing.T) {
	t.Setenv("USER_SCOPE_NAMESPACE_SECRET", "test-secret")
	looper := supportedStickyDecision(t, "looper", config.ToolSelectionModeAdd)
	looper.Algorithm = &config.AlgorithmConfig{Type: config.DecisionAlgorithmReMoM}
	cases := map[string]*config.RouterConfig{
		"without trusted facts": {IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{stickyEnabledDecision(t, "d1")},
		}},
		"looper algorithm": {IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{looper},
		}},
		"redis store": {
			IntelligentRouting: config.IntelligentRouting{
				Decisions: []config.Decision{supportedStickyDecision(t, "d1", config.ToolSelectionModeAdd)},
			},
			ToolSessions: &config.ToolSessionStoreConfig{
				Backend: config.ToolSessionStoreBackendRedis,
				Redis:   &config.ToolSessionRedisConfig{Address: "127.0.0.1:6379"},
			},
		},
	}
	for name, cfg := range cases {
		t.Run(name, func(t *testing.T) {
			_, err := buildOpenAIRouterFromConfig(cfg)
			if !errors.Is(err, config.ErrToolSelectionStickyUnsupported) {
				t.Fatalf("error = %v, want ErrToolSelectionStickyUnsupported", err)
			}
		})
	}
}
