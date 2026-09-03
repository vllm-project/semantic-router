package config

import "fmt"

func (c *ToolsPluginConfig) Validate() error {
	if c == nil || !c.Enabled {
		return nil
	}

	mode := c.EffectiveMode()
	if err := validateToolsPluginMode(mode); err != nil {
		return err
	}
	if err := validateToolsPluginLists(mode, c.AllowTools, c.BlockTools); err != nil {
		return err
	}
	if c.StripToolHistory && mode != ToolsPluginModeNone {
		return fmt.Errorf("tools plugin: strip_tool_history requires mode=%q", ToolsPluginModeNone)
	}

	return c.DynamicRetrieval.Validate()
}

func validateToolsPluginMode(mode string) error {
	switch mode {
	case ToolsPluginModeNone, ToolsPluginModePassthrough, ToolsPluginModeFiltered:
		return nil
	default:
		return fmt.Errorf("tools plugin: mode must be one of %q, %q, or %q", ToolsPluginModeNone, ToolsPluginModePassthrough, ToolsPluginModeFiltered)
	}
}

func validateToolsPluginLists(mode string, allowTools, blockTools []string) error {
	hasLists := len(allowTools) > 0 || len(blockTools) > 0
	switch mode {
	case ToolsPluginModeFiltered:
		if !hasLists {
			return fmt.Errorf("tools plugin: mode=%q requires allow_tools or block_tools", ToolsPluginModeFiltered)
		}
	default:
		if hasLists {
			return fmt.Errorf("tools plugin: allow_tools/block_tools require mode=%q", ToolsPluginModeFiltered)
		}
	}
	return nil
}

// Validate checks the dynamic_retrieval block for legal strategy names and
// numeric ranges.  When the receiver is nil or Enabled is false, validation
// is a no-op so that adding the block to a config never breaks existing
// deployments that leave it disabled.
func (d *DynamicRetrievalConfig) Validate() error {
	if d == nil || !d.Enabled {
		return nil
	}

	switch d.Strategy {
	case "", DynamicRetrievalStrategySemanticOnly, DynamicRetrievalStrategyHybridHistory:
	default:
		return fmt.Errorf("tools plugin: dynamic_retrieval.strategy must be one of %q or %q",
			DynamicRetrievalStrategySemanticOnly, DynamicRetrievalStrategyHybridHistory)
	}

	if d.EffectiveStrategy() == DynamicRetrievalStrategyHybridHistory && d.HistoryWindow < 1 {
		return fmt.Errorf("tools plugin: dynamic_retrieval.history_window must be >= 1 when strategy=%q",
			DynamicRetrievalStrategyHybridHistory)
	}

	if d.MinHistoryConfidence < 0.0 || d.MinHistoryConfidence > 1.0 {
		return fmt.Errorf("tools plugin: dynamic_retrieval.min_history_confidence must be between 0.0 and 1.0")
	}

	if d.Weights != nil {
		if err := d.Weights.Validate(); err != nil {
			return err
		}
	}

	return nil
}

// Validate ensures each weight is non-negative.  Zero weights are permitted
// and mean the corresponding signal is ignored by the retriever.
func (w *DynamicRetrievalWeights) Validate() error {
	if w == nil {
		return nil
	}
	for name, val := range map[string]float64{
		"semantic":           w.Semantic,
		"history":            w.History,
		"decision_prior":     w.DecisionPrior,
		"repetition_penalty": w.RepetitionPenalty,
	} {
		if val < 0.0 {
			return fmt.Errorf("tools plugin: dynamic_retrieval.weights.%s must be non-negative", name)
		}
	}
	return nil
}
