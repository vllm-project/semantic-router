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
	recipe := defaultRecipe(canonical, cfg)

	modelsJSON, err := json.Marshal(modelsProjection{ProviderModels: canonical.Providers.Models})
	if err != nil {
		return nil, fmt.Errorf("configprojection: marshal models: %w", err)
	}

	signalsJSON, err := json.Marshal(recipe.Signals)
	if err != nil {
		return nil, fmt.Errorf("configprojection: marshal signals: %w", err)
	}

	decisionsJSON, err := json.Marshal(recipe.Decisions)
	if err != nil {
		return nil, fmt.Errorf("configprojection: marshal decisions: %w", err)
	}

	pluginsJSON, err := json.Marshal(buildPluginsProjection(canonical, recipe))
	if err != nil {
		return nil, fmt.Errorf("configprojection: marshal plugins: %w", err)
	}

	projectionsJSON, err := json.Marshal(recipe.Projections)
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

// defaultRecipe is the recipe the projection describes: the default recipe.
func defaultRecipe(canonical routerconfig.CanonicalConfig, cfg *routerconfig.RouterConfig) routerconfig.CanonicalRecipe {
	name := string(cfg.DefaultRecipe().Name)
	for _, recipe := range canonical.Recipes {
		if recipe.Name == name {
			return recipe
		}
	}
	return routerconfig.CanonicalRecipe{Name: name}
}

func buildPluginsProjection(canonical routerconfig.CanonicalConfig, recipe routerconfig.CanonicalRecipe) pluginsProjection {
	entries := make([]decisionPluginsEntry, 0, len(recipe.Decisions))
	for _, decision := range recipe.Decisions {
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
