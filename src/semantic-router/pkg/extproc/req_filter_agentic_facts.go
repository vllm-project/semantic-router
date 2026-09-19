package extproc

import (
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agenticfacts"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// agenticFactsEnabled reports whether this deployment ingests the
// caller-presented selection-facts envelope. With the contract disabled the
// router never reads the carrier or trust-marker header, and every other
// function in this file is a no-op.
func (r *OpenAIRouter) agenticFactsEnabled() bool {
	if r == nil || r.Config == nil {
		return false
	}
	return r.Config.AgenticFacts.Enabled
}

// agenticFactsHeaderNames returns the resolved carrier and trust-marker
// header names for this deployment, or nil when the contract is disabled.
// Callers use this list to strip both headers from the outgoing request
// regardless of validation outcome, so a client-declared or gateway-declared
// envelope never reaches the upstream backend.
func (r *OpenAIRouter) agenticFactsHeaderNames() []string {
	if !r.agenticFactsEnabled() {
		return nil
	}
	cfg := r.Config.AgenticFacts
	return []string{cfg.GetCarrierHeader(), cfg.Trust.GetMarkerHeader()}
}

// ingestAgenticFacts reads the caller-presented selection-facts envelope from
// ctx.Headers, enforces the trust boundary, and populates ctx.AgenticFacts.
//
// It never mutates selection. It only records what was accepted or why it
// was rejected, for signal projection and Replay diagnostics to consume
// later. The trust marker is an operator-declared assertion, not a
// cryptographic proof: this function trusts it exactly as far as the
// deployment's gateway is trusted to set and strip it.
func (r *OpenAIRouter) ingestAgenticFacts(ctx *RequestContext) {
	if ctx == nil || !r.agenticFactsEnabled() {
		return
	}
	cfg := r.Config.AgenticFacts

	trustHeader := cfg.Trust.GetMarkerHeader()
	carrierHeader := cfg.GetCarrierHeader()

	trusted := strings.EqualFold(
		strings.TrimSpace(headerValueCI(ctx, trustHeader)),
		cfg.Trust.GetMarkerValue(),
	)
	raw := headerValueCI(ctx, carrierHeader)

	// Both headers are internal to the router. Scrub them from the in-memory
	// context immediately after reading so no later signal or plugin that
	// scans ctx.Headers generically can observe them, independent of the
	// separate HeaderMutation that strips them from the outgoing request.
	removeHeaderValueCI(ctx, trustHeader)
	removeHeaderValueCI(ctx, carrierHeader)

	if !trusted {
		// An untrusted caller's envelope is never parsed, not even to report
		// it as malformed. Presence alone reveals nothing about its content.
		ctx.AgenticFacts = agenticfacts.Result{
			Rejections: []agenticfacts.Rejection{{Reason: agenticfacts.ReasonUntrusted}},
		}
		return
	}
	if raw == "" {
		return // trusted, but nothing presented; zero-value Result is correct
	}

	ctx.AgenticFacts = agenticfacts.Validate(
		[]byte(raw),
		agenticFactsBoundsFromConfig(cfg.Bounds),
		time.Now(),
	)
}

// agenticFactsBoundsFromConfig converts the operator-facing config bounds
// into the validator's own Bounds type. pkg/config cannot import
// pkg/agenticfacts, so this conversion lives here instead.
//
// ClockSkew needs a negative sentinel, not zero, when the operator leaves it
// unset: Bounds.withDefaults treats zero as the deliberate choice to disable
// skew tolerance, and only substitutes the package default for a negative
// value. Passing zero for "unset" would silently disable clock-skew
// tolerance on every deployment that never mentions the field. Every other
// bound treats zero and unset identically, so no sentinel is needed there.
func agenticFactsBoundsFromConfig(cfg config.AgenticFactsBoundsConfig) agenticfacts.Bounds {
	lifetime, _ := cfg.MaxLifetimeDuration() // shape already checked at config load

	skew := time.Duration(-1)
	if cfg.ClockSkewSet() {
		skew, _ = cfg.ClockSkewDuration()
	}

	return agenticfacts.Bounds{
		MaxEnvelopeBytes: cfg.MaxEnvelopeBytes,
		MaxDepth:         cfg.MaxDepth,
		MaxCapabilities:  cfg.MaxCapabilities,
		MaxCandidates:    cfg.MaxCandidates,
		MaxStringLength:  cfg.MaxStringLength,
		MaxLifetime:      lifetime,
		ClockSkew:        skew,
	}
}
