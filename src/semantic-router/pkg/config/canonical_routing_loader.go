package config

import (
	"fmt"

	"gopkg.in/yaml.v2"
)

type routingFragmentDocument struct {
	Routing CanonicalRouting `yaml:"routing"`
}

// ParseRoutingYAMLBytes parses a routing-only YAML fragment containing the DSL-owned
// canonical surface under `routing:`. This intentionally skips provider/global
// cross-reference checks so DSL compile/decompile flows can round-trip fragments.
func ParseRoutingYAMLBytes(data []byte) (*RouterConfig, error) {
	doc := &routingFragmentDocument{}
	if err := yaml.Unmarshal(data, doc); err != nil {
		return nil, fmt.Errorf("failed to parse routing fragment: %w", err)
	}

	cfg := DefaultGlobalConfig()
	cfg.Decisions = copyDecisions(doc.Routing.Decisions)
	ensureModelRefDefaults(cfg.Decisions)
	cfg.Signals = normalizeSignals(doc.Routing.Signals, cfg.Decisions)
	cfg.ModelConfig = make(map[string]ModelParams)

	for _, model := range canonicalRoutingModels(doc.Routing) {
		cfg.ModelConfig[model.Name] = ModelParams{
			ParamSize:         model.ParamSize,
			ContextWindowSize: model.ContextWindowSize,
			Description:       model.Description,
			Capabilities:      append([]string(nil), model.Capabilities...),
			LoRAs:             copyLoRAAdapters(model.LoRAs),
			Tags:              append([]string(nil), model.Tags...),
			QualityScore:      model.QualityScore,
			Modality:          model.Modality,
		}
	}

	if cfg.VectorStore != nil {
		cfg.VectorStore.ApplyDefaults()
	}

	return &cfg, nil
}
