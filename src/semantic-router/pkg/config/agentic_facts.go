package config

import (
	"strings"
	"time"
)

// Defaults applied when an operator leaves the corresponding field unset.
const (
	defaultAgenticFactsCarrierHeader = "x-vsr-agentic-facts"
	defaultAgenticFactsTrustHeader   = "x-vsr-agentic-facts-trusted"
	defaultAgenticFactsTrustValue    = "1"
)

// AgenticFactsConfig controls ingestion of the bounded selection-facts envelope
// that external agent runtimes present at the request boundary.
//
// It is disabled by default. With Enabled false the router never reads the
// carrier header, emits no diagnostics, and routes byte-identically to a build
// without the contract.
type AgenticFactsConfig struct {
	Enabled       bool                     `yaml:"enabled"`
	CarrierHeader string                   `yaml:"carrier_header,omitempty"`
	Trust         AgenticFactsTrustConfig  `yaml:"trust,omitempty"`
	Bounds        AgenticFactsBoundsConfig `yaml:"bounds,omitempty"`
}

// AgenticFactsTrustConfig declares how the router recognizes an envelope that
// arrived through an authenticated gateway rather than directly from a client.
//
// The marker is operator-declared, not cryptographically verified: ExtProc
// cannot tell who set a header. The deployment is responsible for ensuring the
// gateway sets this marker and strips it from client-supplied requests.
// Cryptographic integrity is deferred behind a future policy field.
type AgenticFactsTrustConfig struct {
	MarkerHeader string `yaml:"marker_header,omitempty"`
	MarkerValue  string `yaml:"marker_value,omitempty"`
}

// AgenticFactsBoundsConfig overrides the validator's caps. Every field is
// optional, and an unset field keeps the validator's own default rather than
// meaning "unlimited".
//
// Durations are strings parsed with time.ParseDuration, matching
// provider_reliability.go.
type AgenticFactsBoundsConfig struct {
	MaxEnvelopeBytes int    `yaml:"max_envelope_bytes,omitempty"`
	MaxDepth         int    `yaml:"max_depth,omitempty"`
	MaxCapabilities  int    `yaml:"max_capabilities,omitempty"`
	MaxStringLength  int    `yaml:"max_string_length,omitempty"`
	MaxLifetime      string `yaml:"max_lifetime,omitempty"`
	ClockSkew        string `yaml:"clock_skew,omitempty"`
}

// GetCarrierHeader returns the configured carrier header name, or the default
// when unset.
func (c AgenticFactsConfig) GetCarrierHeader() string {
	if header := strings.TrimSpace(c.CarrierHeader); header != "" {
		return header
	}
	return defaultAgenticFactsCarrierHeader
}

// GetMarkerHeader returns the configured trust marker header, or the default
// when unset.
func (t AgenticFactsTrustConfig) GetMarkerHeader() string {
	if header := strings.TrimSpace(t.MarkerHeader); header != "" {
		return header
	}
	return defaultAgenticFactsTrustHeader
}

// GetMarkerValue returns the value the trust marker header must carry, or the
// default when unset.
func (t AgenticFactsTrustConfig) GetMarkerValue() string {
	if value := strings.TrimSpace(t.MarkerValue); value != "" {
		return value
	}
	return defaultAgenticFactsTrustValue
}

// MaxLifetimeDuration parses MaxLifetime. An unset value yields zero, which the
// caller maps onto the validator's own default.
func (b AgenticFactsBoundsConfig) MaxLifetimeDuration() (time.Duration, error) {
	return parseOptionalDuration(b.MaxLifetime)
}

// ClockSkewDuration parses ClockSkew. Use ClockSkewSet to tell an unset value
// from an explicit zero.
func (b AgenticFactsBoundsConfig) ClockSkewDuration() (time.Duration, error) {
	return parseOptionalDuration(b.ClockSkew)
}

// ClockSkewSet reports whether the operator declared a clock skew at all.
//
// Clock skew is the one bound where zero and unset mean different things: an
// unset value keeps the validator's default tolerance, while an explicit "0s"
// deliberately disables tolerance. Every other bound treats zero as unset.
func (b AgenticFactsBoundsConfig) ClockSkewSet() bool {
	return strings.TrimSpace(b.ClockSkew) != ""
}

// parseOptionalDuration parses a duration string, treating empty as zero with
// no error so callers can apply their own default.
func parseOptionalDuration(value string) (time.Duration, error) {
	trimmed := strings.TrimSpace(value)
	if trimmed == "" {
		return 0, nil
	}
	return time.ParseDuration(trimmed)
}
