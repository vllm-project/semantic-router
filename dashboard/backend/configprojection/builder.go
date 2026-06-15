package configprojection

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"time"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type builtSnapshot struct {
	Version     string
	Source      string
	CreatedAt   time.Time
	DSLSnapshot string
	YAMLHash    string
	Validation  ValidationSummary
	Models      json.RawMessage
	Signals     json.RawMessage
	Decisions   json.RawMessage
	Plugins     json.RawMessage
	Projections json.RawMessage
}

type modelsProjection struct {
	ProviderModels []routerconfig.CanonicalProviderModel `json:"provider_models"`
	ModelCards     []routerconfig.RoutingModel           `json:"model_cards"`
}

type pluginsProjection struct {
	ByDecision []decisionPluginsEntry        `json:"by_decision"`
	Global     *routerconfig.CanonicalGlobal `json:"global,omitempty"`
}

type decisionPluginsEntry struct {
	Decision string                        `json:"decision"`
	Plugins  []routerconfig.DecisionPlugin `json:"plugins"`
}

// BuildSnapshot derives a deployment projection from validated canonical YAML bytes.
func BuildSnapshot(input RefreshInput) (*builtSnapshot, error) {
	if len(input.YAMLBytes) == 0 {
		return nil, fmt.Errorf("configprojection: empty canonical YAML")
	}

	cfg, err := routerconfig.ParseYAMLBytes(input.YAMLBytes)
	if err != nil {
		return nil, fmt.Errorf("configprojection: parse canonical YAML: %w", err)
	}

	canonical := routerconfig.CanonicalConfigFromRouterConfig(cfg)

	modelsJSON, err := json.Marshal(modelsProjection{
		ProviderModels: canonical.Providers.Models,
		ModelCards:     canonical.Routing.ModelCards,
	})
	if err != nil {
		return nil, fmt.Errorf("configprojection: marshal models: %w", err)
	}

	signalsJSON, err := json.Marshal(canonical.Routing.Signals)
	if err != nil {
		return nil, fmt.Errorf("configprojection: marshal signals: %w", err)
	}

	decisionsJSON, err := json.Marshal(canonical.Routing.Decisions)
	if err != nil {
		return nil, fmt.Errorf("configprojection: marshal decisions: %w", err)
	}

	pluginsJSON, err := json.Marshal(buildPluginsProjection(canonical))
	if err != nil {
		return nil, fmt.Errorf("configprojection: marshal plugins: %w", err)
	}

	projectionsJSON, err := json.Marshal(canonical.Routing.Projections)
	if err != nil {
		return nil, fmt.Errorf("configprojection: marshal projections: %w", err)
	}

	version := input.Version
	if version == "" {
		version = time.Now().UTC().Format("20060102-150405")
	}

	source := input.Source
	if source == "" {
		source = SourceManual
	}

	return &builtSnapshot{
		Version:     version,
		Source:      source,
		CreatedAt:   time.Now().UTC(),
		DSLSnapshot: input.DSLSnapshot,
		YAMLHash:    hashYAML(input.YAMLBytes),
		Validation: ValidationSummary{
			Status: "ok",
		},
		Models:      modelsJSON,
		Signals:     signalsJSON,
		Decisions:   decisionsJSON,
		Plugins:     pluginsJSON,
		Projections: projectionsJSON,
	}, nil
}

func buildPluginsProjection(canonical routerconfig.CanonicalConfig) pluginsProjection {
	entries := make([]decisionPluginsEntry, 0, len(canonical.Routing.Decisions))
	for _, decision := range canonical.Routing.Decisions {
		if len(decision.Plugins) == 0 {
			continue
		}
		entries = append(entries, decisionPluginsEntry{
			Decision: decision.Name,
			Plugins:  decision.Plugins,
		})
	}

	return pluginsProjection{
		ByDecision: entries,
		Global:     canonical.Global,
	}
}

func hashYAML(yamlBytes []byte) string {
	sum := sha256.Sum256(yamlBytes)
	return hex.EncodeToString(sum[:])
}
