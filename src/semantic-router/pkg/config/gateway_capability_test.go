package config_test

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const envoyOnlyListener = "needs-envoy"

func init() {
	config.GatewayCapabilities.MustRegister("test_envoy_only_listener", config.GatewayCapability{
		Modes: []config.GatewayMode{config.GatewayExtProc},
		Uses: func(cfg *config.RouterConfig) []config.CapabilityUse {
			var uses []config.CapabilityUse
			for _, listener := range cfg.Listeners {
				if listener.Name == envoyOnlyListener {
					uses = append(uses, config.CapabilityUse{
						Path: "listeners[" + listener.Name + "]", Subject: "listener '" + listener.Name + "'",
					})
				}
			}
			return uses
		},
	})
}

func TestACapabilityRegisteredOutsideConfigPointsToTheModeThatServesIt(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.Listeners = []config.Listener{{Name: envoyOnlyListener, Address: "127.0.0.1", Port: 8899}}
	violations := config.CheckGatewayCapabilities(cfg, config.GatewayStandalone)
	if len(violations) != 1 || violations[0].Capability != "test_envoy_only_listener" ||
		violations[0].Path != "listeners[needs-envoy]" ||
		!strings.Contains(violations[0].Message, "not supported in standalone mode; serve with --gateway extproc") {
		t.Fatalf("violations = %+v", violations)
	}
	if err := config.ValidateGatewayCapabilities(cfg, config.GatewayExtProc); err != nil {
		t.Fatalf("Envoy serves the capability: %v", err)
	}
	if err := config.ValidateGatewayCapabilities(nil, config.GatewayStandalone); err != nil {
		t.Fatalf("a nil configuration uses nothing: %v", err)
	}
}

func TestParseGatewayModeAcceptsTheRoutersModesOnly(t *testing.T) {
	for _, name := range []string{"standalone", "extproc"} {
		if mode, err := config.ParseGatewayMode(name); err != nil || string(mode) != name {
			t.Fatalf("ParseGatewayMode(%q) = %q, %v", name, mode, err)
		}
	}
	for _, name := range []string{"native", "envoy", ""} {
		if _, err := config.ParseGatewayMode(name); err == nil {
			t.Fatalf("ParseGatewayMode(%q) succeeded", name)
		}
	}
}

func TestRoutingDecisionsAtNamesRecipePaths(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.Recipes = []config.RoutingRecipe{
		{Name: config.DefaultRecipeName, Profile: config.RoutingProfile{Decisions: []config.Decision{{Name: "a"}}}},
		{Name: "coding", Profile: config.RoutingProfile{Decisions: []config.Decision{{Name: "b"}}}},
	}
	at := cfg.RoutingDecisionsAt()
	if len(at) != 2 || at[0].Path != "routing.decisions[a]" || at[1].Path != "recipes[coding].routing.decisions[b]" ||
		at[1].Decision.Name != "b" {
		t.Fatalf("RoutingDecisionsAt() = %+v", at)
	}
}
