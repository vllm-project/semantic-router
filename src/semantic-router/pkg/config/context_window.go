package config

import (
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tokencount/tokenizerspec"
)

// Defaults and bounds of global.router.context_window.
const (
	DefaultContextWindowMargin = 0.03
	MaxContextWindowMargin     = 0.5
	DefaultProxyTokenizer      = tokenizerspec.O200kBase
)

// ContextWindowConfig governs the check of a prompt against the context
// windows of the models that can serve it, which runs before the signals.
// The Router counts the prompt itself, with the tokenizer each model names
// in providers.models[].tokenizer or with ProxyTokenizer, and only for a
// prompt whose UTF-8 bytes could exceed a window.
type ContextWindowConfig struct {
	// Margin is the band around a window, as a fraction of it, in which a
	// count decides nothing: at or below window*(1-margin) a prompt fits, at
	// or above window*(1+margin) it is refused, and in between the backend
	// decides. Defaults to 0.03.
	Margin *float64 `yaml:"margin,omitempty" jsonschema:"minimum=0,maximum=0.5"`
	// ProxyTokenizer counts the prompts of models that name no tokenizer: a
	// built-in encoding, a tokenizer.json path or a Hugging Face repo id.
	// Defaults to o200k_base.
	ProxyTokenizer string `yaml:"proxy_tokenizer,omitempty"`
}

// EffectiveMargin is Margin, or its default.
func (c ContextWindowConfig) EffectiveMargin() float64 {
	if c.Margin == nil {
		return DefaultContextWindowMargin
	}
	return *c.Margin
}

// EffectiveProxyTokenizer is ProxyTokenizer, or its default.
func (c ContextWindowConfig) EffectiveProxyTokenizer() string {
	if tokenizer := strings.TrimSpace(c.ProxyTokenizer); tokenizer != "" {
		return tokenizer
	}
	return DefaultProxyTokenizer
}

func validateContextWindowContracts(cfg *RouterConfig) error {
	window := cfg.ContextWindow
	if margin := window.EffectiveMargin(); margin < 0 || margin > MaxContextWindowMargin {
		return fmt.Errorf("global.router.context_window.margin must be between 0 and %g, got %g", MaxContextWindowMargin, margin)
	}
	if window.ProxyTokenizer != "" {
		if _, err := tokenizerspec.Parse(window.ProxyTokenizer); err != nil {
			return fmt.Errorf("global.router.context_window.proxy_tokenizer: %w", err)
		}
	}
	return nil
}
