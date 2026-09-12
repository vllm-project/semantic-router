package config

import "fmt"

const SignalTypeSafety = "safety"

// SafetyRule observes content hazards independently of prompt attacks. Model
// names identify sequence classifiers with complete label distributions.
type SafetyRule struct {
	Name         string            `yaml:"name"`
	Description  string            `yaml:"description,omitempty"`
	Model        string            `yaml:"model,omitempty"`
	Labels       []string          `yaml:"labels,omitempty"`
	UnsafeLabels []string          `yaml:"unsafe_labels,omitempty"`
	Threshold    float64           `yaml:"threshold"`
	Hazard       *SafetyHazardRule `yaml:"hazard,omitempty"`
}

// SafetyHazardRule optionally narrows an unsafe verdict to selected categories.
// Labels describes a complete multi-label sigmoid head. Categories selects
// the hazards of interest; any selected score reaching Threshold matches.
type SafetyHazardRule struct {
	Model      string   `yaml:"model,omitempty"`
	Labels     []string `yaml:"labels"`
	Categories []string `yaml:"categories"`
	Threshold  float64  `yaml:"threshold"`
}

func (r SafetyRule) EffectiveLabels() []string {
	if len(r.Labels) == 0 {
		return []string{"safe", "unsafe"}
	}
	return r.Labels
}

func (r SafetyRule) EffectiveUnsafeLabels() []string {
	if len(r.UnsafeLabels) == 0 {
		return []string{"unsafe"}
	}
	return r.UnsafeLabels
}

func collectSafetyRuleNames(rules []SafetyRule) map[string]struct{} {
	names := make(map[string]struct{}, len(rules))
	for _, rule := range rules {
		names[rule.Name] = struct{}{}
	}
	return names
}

// SafetyModelsConfig owns the built-in local Safety and Hazard artifacts.
// Rules can name an external classifier instead of using these defaults.
type SafetyModelsConfig struct {
	Safety SequenceHeadModelConfig `yaml:"safety"`
	Hazard SequenceHeadModelConfig `yaml:"hazard"`
}

// SequenceHeadModelConfig defines one native sequence head's deployment budget.
// Zero MaxSequenceLength preserves the conservative 512-token default.
type SequenceHeadModelConfig struct {
	ModelID           string                    `yaml:"model_id,omitempty"`
	ModelRef          string                    `yaml:"model_ref,omitempty"`
	UseCPU            bool                      `yaml:"use_cpu"`
	MaxSequenceLength int                       `yaml:"max_sequence_length,omitempty"`
	Window            *SequenceHeadWindowConfig `yaml:"window,omitempty"`
}

func (c SequenceHeadModelConfig) InputLimit() int {
	if c.MaxSequenceLength == 0 {
		return 512
	}
	return c.MaxSequenceLength
}

// NeedsLocalSafetyHeadForRouting accounts for recipe reachability and external
// head overrides. The two artifacts are provisioned independently.
func (c *RouterConfig) NeedsLocalSafetyHeadForRouting(hazard bool) bool {
	if c == nil {
		return false
	}
	needsHead := func(signals Signals, decisions []Decision, projections Projections) bool {
		if !decisionsUseSignalType(decisions, projections, SignalTypeSafety) {
			return false
		}
		for _, rule := range signals.SafetyRules {
			if !hazard && rule.Model == "" {
				return true
			}
			if hazard && rule.Hazard != nil && rule.Hazard.Model == "" {
				return true
			}
		}
		return false
	}
	if c.RoutingScope != "" {
		return needsHead(c.Signals, c.Decisions, c.Projections)
	}
	for _, recipe := range c.ReachableRoutingRecipes() {
		if recipe != nil && needsHead(recipe.Profile.Signals, recipe.Profile.Decisions, recipe.Profile.Projections) {
			return true
		}
	}
	return false
}

// SequenceHeadWindowConfig enables scanning all content tokens in overlapping
// windows. Omission keeps whole-input inference. Size includes special tokens;
// Overlap counts content tokens. Scores are aggregated only after inference.
type SequenceHeadWindowConfig struct {
	Size    int `yaml:"size"`
	Overlap int `yaml:"overlap"`
}

func (c SequenceHeadModelConfig) ValidateWindow() error {
	if c.Window == nil {
		return nil
	}
	if c.Window.Size <= 0 || c.Window.Size > c.InputLimit() {
		return fmt.Errorf("window.size must be positive and at most max_sequence_length")
	}
	if c.Window.Overlap < 0 || c.Window.Overlap >= c.Window.Size {
		return fmt.Errorf("window.overlap must be nonnegative and smaller than window.size")
	}
	// The native tokenizer additionally checks that special tokens leave
	// enough content room for this overlap when scanning the model.
	return nil
}
