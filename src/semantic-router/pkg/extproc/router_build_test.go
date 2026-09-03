package extproc

import (
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

// TestBuildOpenAIRouterFromConfig_StickyEnabledSecretMissing_FailsBeforeComponentBuild
// covers the actual router-construction entry point, not only the
// narrower validator: a config with sticky enabled and no secret must be
// rejected before buildRouterComponents runs (which would otherwise fail
// for unrelated reasons — missing providers, models, etc. — masking
// whether this specific gate actually fired).
func TestBuildOpenAIRouterFromConfig_StickyEnabledSecretMissing_FailsBeforeComponentBuild(t *testing.T) {
	t.Setenv("USER_SCOPE_NAMESPACE_SECRET", "")
	cfg := &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{stickyEnabledDecision(t, "d1")},
		},
	}
	_, err := buildOpenAIRouterFromConfig(cfg)
	if err == nil {
		t.Fatal("expected buildOpenAIRouterFromConfig to fail when sticky is enabled with no secret configured")
	}
	if !strings.Contains(err.Error(), "USER_SCOPE_NAMESPACE_SECRET") {
		t.Fatalf("error = %q, want it to name USER_SCOPE_NAMESPACE_SECRET", err.Error())
	}
}
