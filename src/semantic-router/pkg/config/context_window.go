package config

import (
	"fmt"
	"net/url"
	"slices"
	"strings"

	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tokencount/tokenizerspec"
)

// Defaults and bounds of global.router.context_window.
const (
	DefaultContextWindowMargin      = 0.03
	MaxContextWindowMargin          = 0.5
	DefaultContextWindowProxyMargin = 0.30
	MaxContextWindowProxyMargin     = 1.0
	DefaultProxyTokenizer           = tokenizerspec.O200kBase
)

// ContextWindowConfig governs the check of a prompt against the context
// windows of the models that can serve it, which starts with the request
// and runs beside the signals. The Router counts the prompt itself, with each
// model's EffectiveTokenizer or with ProxyTokenizer, and only for a prompt
// whose UTF-8 bytes could exceed a window.
type ContextWindowConfig struct {
	// Margin is the band around a window, as a fraction of it, in which a
	// count with the model's own tokenizer decides nothing: at or below
	// window*(1-margin) a prompt fits, at or above window*(1+margin) it is
	// refused, and in between the backend decides. Defaults to 0.03.
	Margin *float64 `yaml:"margin,omitempty" jsonschema:"minimum=0,maximum=0.5"`
	// ProxyMargin is the band for a count with the proxy tokenizer, which
	// can be well off a model's own count: Chinese text counts up to a
	// quarter higher with o200k_base than with some models' tokenizers.
	// Defaults to 0.30.
	ProxyMargin *float64 `yaml:"proxy_margin,omitempty" jsonschema:"minimum=0,maximum=1"`
	// ProxyTokenizer counts the prompts of models that name no tokenizer,
	// or whose tokenizer has not loaded: a built-in encoding, a
	// tokenizer.json or its directory, or a Hugging Face repo id. Defaults
	// to o200k_base.
	ProxyTokenizer string `yaml:"proxy_tokenizer,omitempty"`
}

// EffectiveMargin is Margin, or its default.
func (c ContextWindowConfig) EffectiveMargin() float64 {
	if c.Margin == nil {
		return DefaultContextWindowMargin
	}
	return *c.Margin
}

// EffectiveProxyMargin is ProxyMargin, or its default.
func (c ContextWindowConfig) EffectiveProxyMargin() float64 {
	if c.ProxyMargin == nil {
		return DefaultContextWindowProxyMargin
	}
	return *c.ProxyMargin
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
	if margin := window.EffectiveProxyMargin(); margin < 0 || margin > MaxContextWindowProxyMargin {
		return fmt.Errorf("global.router.context_window.proxy_margin must be between 0 and %g, got %g", MaxContextWindowProxyMargin, margin)
	}
	if window.ProxyTokenizer != "" {
		if _, err := tokenizerspec.Parse(window.ProxyTokenizer); err != nil {
			return fmt.Errorf("global.router.context_window.proxy_tokenizer: %w", err)
		}
	}
	return nil
}

// selfHostedSupportTier is the catalog's support tier of the engines an
// operator runs, such as vLLM and SGLang, which serve a model under the id of
// its Hugging Face repo.
const selfHostedSupportTier = "runtime"

// ContextWindowTokenizers are the tokenizers the context window check counts
// with: the proxy tokenizer and each model's EffectiveTokenizer, each once.
func (cfg *RouterConfig) ContextWindowTokenizers() []string {
	if cfg == nil {
		return nil
	}
	proxy := cfg.ContextWindow.EffectiveProxyTokenizer()
	var tokenizers []string
	for _, params := range cfg.ModelConfig {
		if tokenizer := params.EffectiveTokenizer(); tokenizer != "" && tokenizer != proxy && !slices.Contains(tokenizers, tokenizer) {
			tokenizers = append(tokenizers, tokenizer)
		}
	}
	slices.Sort(tokenizers)
	return append([]string{proxy}, tokenizers...)
}

// hubReservedPaths are the first path segments of huggingface.co pages that
// are not model repos.
var hubReservedPaths = map[string]bool{
	"api": true, "blog": true, "collections": true, "datasets": true, "docs": true,
	"models": true, "organizations": true, "papers": true, "spaces": true,
}

// hubRepoOf is the Hugging Face repo an open-weights distribution comes
// from, by its source https://huggingface.co/<org>/<name>[/...], or "".
func hubRepoOf(distribution modelcatalog.ModelDistribution) string {
	if distribution.Type != "open_weights" {
		return ""
	}
	source, err := url.Parse(distribution.Source)
	if err != nil || source.Scheme != "https" || source.Host != "huggingface.co" || source.User != nil {
		return ""
	}
	parts := strings.Split(strings.Trim(source.Path, "/"), "/")
	if len(parts) < 2 || hubReservedPaths[parts[0]] {
		return ""
	}
	return hubRepoID(parts[0] + "/" + parts[1])
}

// hubRepoID is id when it is a Hugging Face repo id, org/name, else "".
func hubRepoID(id string) string {
	if !strings.Contains(id, "/") || strings.Contains(id, "@") {
		return ""
	}
	if spec, err := tokenizerspec.Parse(id); err != nil || spec.Kind != tokenizerspec.Hub {
		return ""
	}
	return id
}
