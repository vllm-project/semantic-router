package config

import "fmt"

// OnErrorAllow and OnErrorBlock are the values a pluggable classifier
// backend's OnError field accepts. Deliberately NOT named/valued after this
// package's other on_error: skip|fail fields (fusion/remom/workflows/
// confidence configs) - there, "fail" means "propagate the error and halt";
// here, "block" means "treat as a positive detection and close the
// request", a different behavior the same word would misleadingly imply is
// the same. allow/block names the actual effect on the request instead.
const (
	// OnErrorAllow preserves the historical behavior: a classify error is
	// logged and the affected content is treated as not matching, so other
	// content/rules still evaluate normally. This is the default when
	// OnError is unset.
	OnErrorAllow = "allow"
	// OnErrorBlock treats a classify error as if the rule matched at
	// maximum confidence - fail-closed, because an inference failure means
	// the content could not be verified safe. Without this, an unreachable
	// classifier endpoint looks identical to a genuinely clean request (see
	// @adaamko's review on #2760).
	OnErrorBlock = "block"
)

// ClassifierOnErrorConfig is the shared allow|block failure-handling contract
// for a pluggable classifier backend (PromptGuardConfig today; CategoryModel/
// PIIModel/ClassifierSignalRule are expected to embed it too rather than
// each declaring an independent on_error field - see @adaamko's scoping
// question on #2918/#2930: a knob meant to cover every classifier backend
// isn't reachable from the others if it only lives on PromptGuardConfig).
// Embed it with `yaml:",inline"` so `on_error` still decodes at the embedding
// struct's own level.
type ClassifierOnErrorConfig struct {
	// OnError selects what a classifier-backend failure does to the rule
	// that failed to evaluate: OnErrorAllow (default) or OnErrorBlock.
	OnError string `yaml:"on_error,omitempty"`
}

// IsBlock reports whether a classify failure should be treated as a positive
// detection (fail-closed) rather than tolerated.
func (c ClassifierOnErrorConfig) IsBlock() bool {
	return c.OnError == OnErrorBlock
}

// ValidateOnError rejects any OnError value other than the empty default,
// allow, or block.
//
// Deliberately not named Validate: this struct is embedded, so a plain
// Validate would be promoted onto every embedding config (PromptGuardConfig,
// and CanonicalPromptGuardModule at depth 2). A caller reaching for
// promptGuard.Validate() reasonably expects the whole prompt-guard config to
// be checked, but would get a method that only inspects on_error and returns
// nil for a genuinely invalid config (e.g. variant and protocol both set, or
// an unrecognized protocol - both of which only
// validatePromptGuardBackendConfig catches). The narrower name keeps the
// promoted surface honest about what it verifies.
func (c ClassifierOnErrorConfig) ValidateOnError() error {
	switch c.OnError {
	case "", OnErrorAllow, OnErrorBlock:
		return nil
	default:
		return fmt.Errorf("on_error: unrecognized value %q, must be one of: %s, %s", c.OnError, OnErrorAllow, OnErrorBlock)
	}
}

// UnscannedConfig is what a safety rule (prompt guard, PII) does with content
// its model did not read in full: longer than the model's input or scan
// budget, truncated, or not scanned within the signal's deadline. With
// OnErrorBlock (the default) the rule matches it, so padding a request cannot
// carry content past the guard; with OnErrorAllow the content follows
// on_error, as a backend failure does.
type UnscannedConfig struct {
	OnUnscanned string `yaml:"on_unscanned,omitempty"`
}

// UnscannedBlocks reports whether content the model did not read matches the rule.
func (c UnscannedConfig) UnscannedBlocks() bool {
	return c.OnUnscanned != OnErrorAllow
}

// ValidateOnUnscanned rejects any OnUnscanned value other than the empty
// default, allow, or block.
func (c UnscannedConfig) ValidateOnUnscanned() error {
	switch c.OnUnscanned {
	case "", OnErrorAllow, OnErrorBlock:
		return nil
	default:
		return fmt.Errorf("on_unscanned: unrecognized value %q, must be one of: %s, %s", c.OnUnscanned, OnErrorAllow, OnErrorBlock)
	}
}
