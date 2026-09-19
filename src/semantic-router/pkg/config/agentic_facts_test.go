package config

import (
	"strings"
	"testing"
)

func agenticFactsCfg(facts AgenticFactsConfig) *RouterConfig {
	cfg := &RouterConfig{}
	cfg.AgenticFacts = facts
	return cfg
}

func TestValidateAgenticFactsConfigAcceptsAbsentBlock(t *testing.T) {
	if err := validateAgenticFactsConfig(nil); err != nil {
		t.Fatalf("nil config must be accepted: %v", err)
	}
	if err := validateAgenticFactsConfig(agenticFactsCfg(AgenticFactsConfig{})); err != nil {
		t.Fatalf("an omitted agentic_facts block must be accepted: %v", err)
	}
}

// The values shipped in config/config.yaml must validate, or the reference
// config becomes an example operators cannot actually load.
func TestValidateAgenticFactsConfigAcceptsReferenceValues(t *testing.T) {
	facts := AgenticFactsConfig{
		CarrierHeader: "x-vsr-agentic-facts",
		Trust: AgenticFactsTrustConfig{
			MarkerHeader: "x-vsr-agentic-facts-trusted",
			MarkerValue:  "1",
		},
		Bounds: AgenticFactsBoundsConfig{
			MaxEnvelopeBytes: 8192,
			MaxDepth:         16,
			MaxCapabilities:  16,
			MaxCandidates:    32,
			MaxStringLength:  128,
			MaxLifetime:      "5m",
			ClockSkew:        "5s",
		},
	}

	if err := validateAgenticFactsConfig(agenticFactsCfg(facts)); err != nil {
		t.Fatalf("reference config values must validate: %v", err)
	}
}

// A header name carrying a colon or whitespace never matches a real request
// header, so the contract would silently never activate.
func TestValidateAgenticFactsRejectsMalformedHeaderNames(t *testing.T) {
	tests := map[string]AgenticFactsConfig{
		"carrier with colon": {CarrierHeader: "x-facts: value"},
		"carrier with space": {CarrierHeader: "x facts"},
		"carrier with tab":   {CarrierHeader: "x\tfacts"},
		"marker with colon":  {Trust: AgenticFactsTrustConfig{MarkerHeader: "x-trusted: 1"}},
	}

	for name, facts := range tests {
		t.Run(name, func(t *testing.T) {
			if err := validateAgenticFactsConfig(agenticFactsCfg(facts)); err == nil {
				t.Fatal("a header name that cannot match a real header must be rejected")
			}
		})
	}
}

// Whitespace-only values are not malformed: the accessors trim them, so they
// mean the same thing as an omitted field.
func TestValidateAgenticFactsAcceptsWhitespaceOnlyHeaderNames(t *testing.T) {
	facts := AgenticFactsConfig{CarrierHeader: "   "}

	if err := validateAgenticFactsConfig(agenticFactsCfg(facts)); err != nil {
		t.Fatalf("a whitespace-only header must fall back to the default: %v", err)
	}
	if got := facts.GetCarrierHeader(); got != "x-vsr-agentic-facts" {
		t.Fatalf("want the documented default, got %q", got)
	}
}

// If both names resolve to one header, the request carries either the envelope
// or the trust marker but never both, and the contract silently never fires.
func TestValidateAgenticFactsRejectsCollidingHeaders(t *testing.T) {
	tests := map[string]AgenticFactsConfig{
		"both set to the same name": {
			CarrierHeader: "x-shared",
			Trust:         AgenticFactsTrustConfig{MarkerHeader: "x-shared"},
		},
		"same name in different case": {
			CarrierHeader: "X-Shared",
			Trust:         AgenticFactsTrustConfig{MarkerHeader: "x-shared"},
		},
		// Only a comparison of resolved values catches the next two: the raw
		// fields differ, and one of them is empty.
		"carrier set to the marker default, marker unset": {
			CarrierHeader: "x-vsr-agentic-facts-trusted",
		},
		"marker set to the carrier default, carrier unset": {
			Trust: AgenticFactsTrustConfig{MarkerHeader: "x-vsr-agentic-facts"},
		},
	}

	for name, facts := range tests {
		t.Run(name, func(t *testing.T) {
			err := validateAgenticFactsConfig(agenticFactsCfg(facts))
			if err == nil {
				t.Fatal("a carrier and trust marker on one header must be rejected")
			}
			if !strings.Contains(err.Error(), "must differ") {
				t.Fatalf("want a collision error, got %v", err)
			}
		})
	}
}

func TestValidateAgenticFactsRejectsNegativeBounds(t *testing.T) {
	tests := map[string]AgenticFactsBoundsConfig{
		"max_envelope_bytes": {MaxEnvelopeBytes: -1},
		"max_depth":          {MaxDepth: -1},
		"max_capabilities":   {MaxCapabilities: -1},
		"max_candidates":     {MaxCandidates: -1},
		"max_string_length":  {MaxStringLength: -1},
	}

	for name, bounds := range tests {
		t.Run(name, func(t *testing.T) {
			err := validateAgenticFactsConfig(agenticFactsCfg(AgenticFactsConfig{Bounds: bounds}))
			if err == nil {
				t.Fatal("a negative bound must be rejected")
			}
			if !strings.Contains(err.Error(), name) {
				t.Fatalf("error must name the offending field, got %v", err)
			}
		})
	}
}

// Zero means "keep the validator's own default", so a partial bounds block must
// stay legal.
func TestValidateAgenticFactsAcceptsZeroBounds(t *testing.T) {
	facts := AgenticFactsConfig{Bounds: AgenticFactsBoundsConfig{MaxDepth: 8}}

	if err := validateAgenticFactsConfig(agenticFactsCfg(facts)); err != nil {
		t.Fatalf("a partial bounds block must be accepted: %v", err)
	}
}

func TestValidateAgenticFactsRejectsUnparseableDurations(t *testing.T) {
	tests := map[string]AgenticFactsBoundsConfig{
		"max_lifetime in prose": {MaxLifetime: "5 minutes"},
		"max_lifetime garbage":  {MaxLifetime: "soon"},
		"clock_skew in prose":   {ClockSkew: "5 seconds"},
	}

	for name, bounds := range tests {
		t.Run(name, func(t *testing.T) {
			err := validateAgenticFactsConfig(agenticFactsCfg(AgenticFactsConfig{Bounds: bounds}))
			if err == nil {
				t.Fatal("a duration Go cannot parse must be rejected")
			}
		})
	}
}

// A zero or negative lifetime expires every envelope on arrival, which is never
// what an operator means.
func TestValidateAgenticFactsRejectsNonPositiveMaxLifetime(t *testing.T) {
	for _, value := range []string{"0s", "-1m"} {
		t.Run(value, func(t *testing.T) {
			facts := AgenticFactsConfig{Bounds: AgenticFactsBoundsConfig{MaxLifetime: value}}
			if err := validateAgenticFactsConfig(agenticFactsCfg(facts)); err == nil {
				t.Fatal("a non-positive max_lifetime must be rejected")
			}
		})
	}
}

// Clock skew is the one bound where zero is a deliberate choice rather than an
// omission: it disables tolerance instead of restoring the default.
func TestValidateAgenticFactsClockSkewZeroIsValidButNegativeIsNot(t *testing.T) {
	zero := AgenticFactsConfig{Bounds: AgenticFactsBoundsConfig{ClockSkew: "0s"}}
	if err := validateAgenticFactsConfig(agenticFactsCfg(zero)); err != nil {
		t.Fatalf("an explicit zero clock skew must be accepted: %v", err)
	}

	negative := AgenticFactsConfig{Bounds: AgenticFactsBoundsConfig{ClockSkew: "-5s"}}
	if err := validateAgenticFactsConfig(agenticFactsCfg(negative)); err == nil {
		t.Fatal("a negative clock skew would reject envelopes that have not expired")
	}
}

func TestAgenticFactsAccessorsReturnDocumentedDefaults(t *testing.T) {
	var facts AgenticFactsConfig

	if got := facts.GetCarrierHeader(); got != "x-vsr-agentic-facts" {
		t.Errorf("carrier header default: got %q", got)
	}
	if got := facts.Trust.GetMarkerHeader(); got != "x-vsr-agentic-facts-trusted" {
		t.Errorf("marker header default: got %q", got)
	}
	if got := facts.Trust.GetMarkerValue(); got != "1" {
		t.Errorf("marker value default: got %q", got)
	}
}

func TestAgenticFactsAccessorsTrimConfiguredValues(t *testing.T) {
	facts := AgenticFactsConfig{
		CarrierHeader: "  x-custom-facts  ",
		Trust:         AgenticFactsTrustConfig{MarkerHeader: " x-custom-trusted "},
	}

	if got := facts.GetCarrierHeader(); got != "x-custom-facts" {
		t.Errorf("carrier header must be trimmed, got %q", got)
	}
	if got := facts.Trust.GetMarkerHeader(); got != "x-custom-trusted" {
		t.Errorf("marker header must be trimmed, got %q", got)
	}
}

// ClockSkewSet is what lets an explicit "0s" survive conversion into the
// validator's Bounds, where zero would otherwise be read as "unset".
func TestAgenticFactsClockSkewSetDistinguishesUnsetFromZero(t *testing.T) {
	var unset AgenticFactsBoundsConfig
	if unset.ClockSkewSet() {
		t.Error("an omitted clock skew must not report as set")
	}

	explicit := AgenticFactsBoundsConfig{ClockSkew: "0s"}
	if !explicit.ClockSkewSet() {
		t.Error("an explicit zero clock skew must report as set")
	}

	skew, err := explicit.ClockSkewDuration()
	if err != nil || skew != 0 {
		t.Fatalf("want 0 with no error, got %v / %v", skew, err)
	}
}

// MaxLifetimeDuration must report an unset lifetime as zero with no error, so
// callers can apply the validator's own default rather than treating it as a
// configuration failure.
func TestAgenticFactsMaxLifetimeDurationParsesAndDefaults(t *testing.T) {
	var unset AgenticFactsBoundsConfig
	lifetime, err := unset.MaxLifetimeDuration()
	if err != nil || lifetime != 0 {
		t.Fatalf("an unset max_lifetime must yield 0 with no error, got %v / %v", lifetime, err)
	}

	configured := AgenticFactsBoundsConfig{MaxLifetime: " 90s "}
	lifetime, err = configured.MaxLifetimeDuration()
	if err != nil {
		t.Fatalf("a padded duration must parse: %v", err)
	}
	if lifetime.Seconds() != 90 {
		t.Fatalf("want 90s, got %v", lifetime)
	}
}
