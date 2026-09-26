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

// TestValidateStickyToolSelectionPhaseSupport_StickyEnabledSecretConfigured_Err
// covers the maintainer-flagged silent-no-op hazard directly (issue #3347
// phase 1 / sub-issue #3392): sticky.enabled: true must be rejected even
// when USER_SCOPE_NAMESPACE_SECRET is configured — the narrow secret
// validator above accepts this config (correctly, for its own scope), but
// no request path consumes ResolveStickyToolIdentity or the sessiontools
// store yet, so a configured secret alone must not be enough to let sticky
// through as a silent no-op.
func TestValidateStickyToolSelectionPhaseSupport_StickyEnabledSecretConfigured_Err(t *testing.T) {
	t.Setenv("USER_SCOPE_NAMESPACE_SECRET", "test-secret")
	cfg := &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{stickyEnabledDecision(t, "d1")},
		},
	}

	err := validateStickyToolSelectionPhaseSupport(cfg)
	if !errors.Is(err, config.ErrToolSelectionStickyUnsupported) {
		t.Fatalf("error = %v, want ErrToolSelectionStickyUnsupported", err)
	}
}

func TestValidateStickyToolSelectionPhaseSupport_NoStickyDecisions_OK(t *testing.T) {
	cfg := &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{stickyDisabledDecision(t, "d1")},
		},
	}
	if err := validateStickyToolSelectionPhaseSupport(cfg); err != nil {
		t.Fatalf("no decision enables sticky, expected no error, got: %v", err)
	}
}

func TestValidateStickyToolSelectionSecret_NilConfig_OK(t *testing.T) {
	t.Setenv("USER_SCOPE_NAMESPACE_SECRET", "")
	if err := validateStickyToolSelectionSecret(nil); err != nil {
		t.Fatalf("nil config, expected no error, got: %v", err)
	}
}

// TestBuildOpenAIRouterFromConfig_StickyEnabledSecretConfigured_FailsUnsupportedBeforeComponentBuild
// is the direct regression for the maintainer's reported issue (#3392): a
// config with sticky enabled — and, notably, USER_SCOPE_NAMESPACE_SECRET
// *configured* — must still be rejected before buildRouterComponents runs.
// Before this fix, a configured secret was enough to let sticky.enabled:
// true pass both config validation and router construction, even though no
// request path consumed it — a silent no-op. Setting the secret here rules
// that variable out, so a failure can only come from the phase-support
// gate, not the secret gate.
func TestBuildOpenAIRouterFromConfig_StickyEnabledSecretConfigured_FailsUnsupportedBeforeComponentBuild(t *testing.T) {
	t.Setenv("USER_SCOPE_NAMESPACE_SECRET", "test-secret")
	cfg := &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{stickyEnabledDecision(t, "d1")},
		},
	}

	_, err := buildOpenAIRouterFromConfig(cfg)
	if !errors.Is(err, config.ErrToolSelectionStickyUnsupported) {
		t.Fatalf("error = %v, want ErrToolSelectionStickyUnsupported", err)
	}
}
