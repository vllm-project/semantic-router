package config

import (
	"errors"
	"strings"
	"testing"
)

func supportedStickyToolsPlugin() *ToolsPluginConfig {
	return &ToolsPluginConfig{
		Enabled: true,
		Mode:    ToolsPluginModePassthrough,
		TrustedFacts: &TrustedFactsConfig{
			Enabled:      true,
			Enforcement:  TrustedEnforcementAuthoritative,
			TrustSources: []string{TrustedSourceOperatorPolicy},
			StageRoles:   []string{TrustedStageCandidate, TrustedStageFinal},
		},
	}
}

func stickySupportConfig(sticky bool, tools *ToolsPluginConfig, algorithm *AlgorithmConfig, store *ToolSessionStoreConfig) *RouterConfig {
	plugins := []DecisionPlugin{{
		Type: DecisionPluginToolSelection,
		Configuration: MustStructuredPayload(&ToolSelectionPluginConfig{
			Enabled: true, Mode: ToolSelectionModeAdd,
			Sticky: &StickyToolSelectionConfig{Enabled: sticky},
		}),
	}}
	if tools != nil {
		plugins = append(plugins, DecisionPlugin{Type: DecisionPluginTools, Configuration: MustStructuredPayload(tools)})
	}
	return &RouterConfig{
		IntelligentRouting: IntelligentRouting{Decisions: []Decision{{
			Name: "sticky-decision", Algorithm: algorithm, Plugins: plugins,
		}}},
		ToolSessions: store,
	}
}

func TestValidateStickyToolSelectionSupportAcceptsLocalAuthoritativeDecision(t *testing.T) {
	for name, store := range map[string]*ToolSessionStoreConfig{
		"default store":  nil,
		"explicit local": {Backend: ToolSessionStoreBackendLocal},
	} {
		t.Run(name, func(t *testing.T) {
			if err := ValidateStickyToolSelectionSupport(stickySupportConfig(true, supportedStickyToolsPlugin(), nil, store)); err != nil {
				t.Fatalf("supported local sticky decision rejected: %v", err)
			}
		})
	}
}

func TestValidateStickyToolSelectionSupportRejectsUnsupportedRuntime(t *testing.T) {
	withTools := func(edit func(*ToolsPluginConfig)) *ToolsPluginConfig {
		tools := supportedStickyToolsPlugin()
		edit(tools)
		return tools
	}
	redis := &ToolSessionStoreConfig{
		Backend: ToolSessionStoreBackendRedis,
		Redis:   &ToolSessionRedisConfig{Address: "127.0.0.1:6379"},
	}
	cases := map[string]struct {
		tools     *ToolsPluginConfig
		algorithm *AlgorithmConfig
		store     *ToolSessionStoreConfig
		want      string
	}{
		"no tools plugin":        {want: "must enable trusted_facts"},
		"trusted facts disabled": {tools: withTools(func(c *ToolsPluginConfig) { c.TrustedFacts.Enabled = false }), want: "must enable trusted_facts"},
		"disabled parent":        {tools: withTools(func(c *ToolsPluginConfig) { c.Enabled = false }), want: "must enable trusted_facts"},
		"advisory enforcement": {
			tools: withTools(func(c *ToolsPluginConfig) { c.TrustedFacts.Enforcement = TrustedEnforcementAdvisory }),
			want:  "enforcement must be",
		},
		"no authorizing source": {
			tools: withTools(func(c *ToolsPluginConfig) {
				c.TrustedFacts.TrustSources = []string{TrustedSourceGatewayAttested}
			}),
			want: "authorizes tool use",
		},
		"candidate only": {
			tools: withTools(func(c *ToolsPluginConfig) { c.TrustedFacts.StageRoles = []string{TrustedStageCandidate} }),
			want:  `"final"`,
		},
		"mode none":   {tools: withTools(func(c *ToolsPluginConfig) { c.Mode = ToolsPluginModeNone }), want: "removes every tool"},
		"looper":      {tools: supportedStickyToolsPlugin(), algorithm: &AlgorithmConfig{Type: DecisionAlgorithmReMoM}, want: "several models"},
		"redis store": {tools: supportedStickyToolsPlugin(), store: redis, want: `backend "redis"`},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			err := ValidateStickyToolSelectionSupport(stickySupportConfig(true, tc.tools, tc.algorithm, tc.store))
			if !errors.Is(err, ErrToolSelectionStickyUnsupported) || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error = %v, want ErrToolSelectionStickyUnsupported mentioning %q", err, tc.want)
			}
		})
	}
}

// Disabled sticky blocks stay inert: no trusted-facts requirement and no
// rejection of a configured Redis store.
func TestValidateStickyToolSelectionSupportIgnoresDisabledSticky(t *testing.T) {
	redis := &ToolSessionStoreConfig{
		Backend: ToolSessionStoreBackendRedis,
		Redis:   &ToolSessionRedisConfig{Address: "127.0.0.1:6379"},
	}
	if err := ValidateStickyToolSelectionSupport(stickySupportConfig(false, nil, nil, redis)); err != nil {
		t.Fatalf("disabled sticky must stay inert, got %v", err)
	}
}

// The supported-runtime check runs on the admission path for every recipe.
func TestStickyToolSelectionSupportRunsAtAdmission(t *testing.T) {
	cfg := stickySupportConfig(true, supportedStickyToolsPlugin(), &AlgorithmConfig{Type: DecisionAlgorithmReMoM}, nil)
	err := validateConfigContracts(cfg)
	if !errors.Is(err, ErrToolSelectionStickyUnsupported) || !strings.Contains(err.Error(), "decision 'sticky-decision'") {
		t.Fatalf("admission error = %v, want unsupported Looper sticky decision", err)
	}
}
