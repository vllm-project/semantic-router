package config

import (
	"fmt"
	"math"
)

// Defaults and bounds of a decision deployment's routing view and scan
// window. A remote System One API takes longer the more it reads and its
// calls run under a short timeout, so a systemone deployment reads less per
// call by default.
const (
	DefaultRoutingViewTokens          = 8192
	DefaultSystemOneRoutingViewTokens = 4096
	DefaultRoutingViewHeadShare       = 0.25
	DefaultScanWindowTokens           = 8192
	DefaultScanWindowOverlap          = 512
	DefaultSystemOneScanWindowTokens  = 4096
	DefaultSystemOneScanWindowOverlap = 256
	DefaultScanWindowParallel         = 4
	MaxScanWindowParallel             = 64
	MinModelInputTokens               = 256
	MaxModelInputTokens               = 1 << 20
)

// DeploymentRoutingView bounds what a routing question sends a decision
// deployment about a long input. A routing question is a rule that does not
// read its whole input (coverage truncate, every application preset but the
// guards). An input longer than the view is read as its head and its tail
// around a marker that says how much was left out; an input within it goes
// as it is. The view is sized in bytes: four a token of MaxTokens, about
// what prose takes, and never more than the model's validated length less
// the question, a byte a token, so dense text cannot overrun the model.
type DeploymentRoutingView struct {
	// MaxTokens is the tokens of prose a routing question reads at most, as
	// four bytes each (default 8192; 4096 for a systemone deployment).
	MaxTokens int `yaml:"max_tokens,omitempty" json:"max_tokens,omitempty" jsonschema:"minimum=256,maximum=1048576"`
	// HeadShare is the part of the view the head takes; the tail takes the
	// rest (default 0.25).
	HeadShare *float64 `yaml:"head_share,omitempty" json:"head_share,omitempty" jsonschema:"minimum=0,maximum=1"`
}

// DeploymentScanWindow sizes the windows a guard rule (jailbreak, safety,
// pii) reads a long input in: windows of Tokens, each overlapping the one
// before by Overlap, read tail first, then head, then the middle, each in a
// call of its own with at most MaxParallel in flight. Windows are sized in
// bytes as routing views are, so the model's validated length bounds a
// window whatever Tokens says and however dense the text.
type DeploymentScanWindow struct {
	// Tokens is the size of one window in tokens of prose, as four bytes
	// each (default 8192; 4096 for a systemone deployment).
	Tokens int `yaml:"tokens,omitempty" json:"tokens,omitempty" jsonschema:"minimum=256,maximum=1048576"`
	// Overlap is what two neighbouring windows share, so text across a cut
	// is read whole by one of them (default 512; 256 for a systemone
	// deployment); less than half a window.
	Overlap *int `yaml:"overlap,omitempty" json:"overlap,omitempty" jsonschema:"minimum=0"`
	// MaxParallel is how many windows of a request's guard scans the
	// deployment reads at a time (default 4). The windows read when the
	// signals' deadline ends are the ones a partial scan decides on.
	MaxParallel int `yaml:"max_parallel,omitempty" json:"max_parallel,omitempty" jsonschema:"minimum=1,maximum=64"`
}

// RoutingViewTokens is the routing view's bound and head share the
// deployment reads, its own or its provider's default.
func (d ModelDeployment) RoutingViewTokens() (int, float64) {
	tokens, share := DefaultRoutingViewTokens, DefaultRoutingViewHeadShare
	if d.IsSystemOne() {
		tokens = DefaultSystemOneRoutingViewTokens
	}
	if view := d.RoutingView; view != nil {
		if view.MaxTokens > 0 {
			tokens = view.MaxTokens
		}
		if view.HeadShare != nil {
			share = *view.HeadShare
		}
	}
	return tokens, share
}

// ScanWindowTokens is the size and overlap of the deployment's scan
// windows, its own or its provider's default.
func (d ModelDeployment) ScanWindowTokens() (int, int) {
	tokens, overlap := DefaultScanWindowTokens, DefaultScanWindowOverlap
	if d.IsSystemOne() {
		tokens, overlap = DefaultSystemOneScanWindowTokens, DefaultSystemOneScanWindowOverlap
	}
	if window := d.ScanWindow; window != nil {
		if window.Tokens > 0 {
			tokens = window.Tokens
		}
		if window.Overlap != nil {
			overlap = *window.Overlap
		}
	}
	return tokens, min(overlap, tokens/2-1)
}

// ScanWindowParallel is how many of a request's guard windows the
// deployment reads at a time.
func (d ModelDeployment) ScanWindowParallel() int {
	if window := d.ScanWindow; window != nil && window.MaxParallel > 0 {
		return window.MaxParallel
	}
	return DefaultScanWindowParallel
}

func (d ModelDeployment) validateInputViews() error {
	if d.RoutingView == nil && d.ScanWindow == nil {
		return nil
	}
	if !d.IsModelRuntime() && !d.IsSystemOne() {
		return fmt.Errorf("routing_view and scan_window bound what a decision model reads; they apply to model_runtime and systemone deployments")
	}
	if view := d.RoutingView; view != nil {
		if view.MaxTokens != 0 && (view.MaxTokens < MinModelInputTokens || view.MaxTokens > MaxModelInputTokens) {
			return fmt.Errorf("routing_view.max_tokens must be between %d and %d", MinModelInputTokens, MaxModelInputTokens)
		}
		if share := view.HeadShare; share != nil && (math.IsNaN(*share) || *share < 0 || *share > 1) {
			return fmt.Errorf("routing_view.head_share must be between 0 and 1")
		}
	}
	if window := d.ScanWindow; window != nil {
		if window.Tokens != 0 && (window.Tokens < MinModelInputTokens || window.Tokens > MaxModelInputTokens) {
			return fmt.Errorf("scan_window.tokens must be between %d and %d", MinModelInputTokens, MaxModelInputTokens)
		}
		tokens, _ := d.ScanWindowTokens()
		if overlap := window.Overlap; overlap != nil && (*overlap < 0 || *overlap >= tokens/2) {
			return fmt.Errorf("scan_window.overlap must be at least 0 and less than half of scan_window.tokens (%d)", tokens)
		}
		if window.MaxParallel < 0 || window.MaxParallel > MaxScanWindowParallel {
			return fmt.Errorf("scan_window.max_parallel must be between 1 and %d", MaxScanWindowParallel)
		}
	}
	return nil
}
