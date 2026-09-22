package config

import (
	"fmt"
	"strings"
)

// validateAgenticFactsConfig checks the agentic facts contract.
//
// Shape checks run even when the contract is disabled, so an operator who flips
// enabled later does not discover a typo at that moment. Nothing here is
// required: every field has a documented default, so an empty block is valid.
func validateAgenticFactsConfig(cfg *RouterConfig) error {
	if cfg == nil {
		return nil
	}
	if err := validateAgenticFactsHeaders(cfg.AgenticFacts); err != nil {
		return err
	}
	return validateAgenticFactsBounds(cfg.AgenticFacts.Bounds)
}

func validateAgenticFactsHeaders(facts AgenticFactsConfig) error {
	headers := []struct {
		path  string
		value string
	}{
		{"global.router.agentic_facts.carrier_header", facts.CarrierHeader},
		{"global.router.agentic_facts.trust.marker_header", facts.Trust.MarkerHeader},
	}
	for _, header := range headers {
		if err := validateAgenticFactsHeaderName(header.path, header.value); err != nil {
			return err
		}
	}

	// The carrier and the trust marker are read independently on every request,
	// so one header cannot answer both questions. Compare the resolved values so
	// that setting one to the other's default is caught too.
	if strings.EqualFold(facts.GetCarrierHeader(), facts.Trust.GetMarkerHeader()) {
		return fmt.Errorf(
			"global.router.agentic_facts.carrier_header and trust.marker_header must differ, both resolve to %q",
			facts.GetCarrierHeader(),
		)
	}
	return nil
}

// validateAgenticFactsHeaderName rejects values that cannot name an HTTP
// header. An empty value is valid and means the documented default applies.
func validateAgenticFactsHeaderName(path string, value string) error {
	trimmed := strings.TrimSpace(value)
	if trimmed == "" {
		return nil
	}
	if strings.ContainsAny(trimmed, " \t:") {
		return fmt.Errorf("%s must be a header name without spaces or colons, got %q", path, value)
	}
	return nil
}

// validateAgenticFactsBounds rejects bounds the validator could not act on.
// Zero is accepted throughout and means "keep the package default"; only
// negative values, unparseable durations, and a non-positive declared lifetime
// are errors.
func validateAgenticFactsBounds(bounds AgenticFactsBoundsConfig) error {
	numeric := []struct {
		name  string
		value int
	}{
		{"max_envelope_bytes", bounds.MaxEnvelopeBytes},
		{"max_depth", bounds.MaxDepth},
		{"max_capabilities", bounds.MaxCapabilities},
		{"max_string_length", bounds.MaxStringLength},
	}
	for _, field := range numeric {
		if field.value < 0 {
			return fmt.Errorf(
				"global.router.agentic_facts.bounds.%s must be >= 0, got %d",
				field.name, field.value,
			)
		}
	}

	lifetime, err := bounds.MaxLifetimeDuration()
	if err != nil {
		return fmt.Errorf("global.router.agentic_facts.bounds.max_lifetime is invalid: %w", err)
	}
	if strings.TrimSpace(bounds.MaxLifetime) != "" && lifetime <= 0 {
		return fmt.Errorf(
			"global.router.agentic_facts.bounds.max_lifetime must be positive, got %q",
			bounds.MaxLifetime,
		)
	}

	skew, err := bounds.ClockSkewDuration()
	if err != nil {
		return fmt.Errorf("global.router.agentic_facts.bounds.clock_skew is invalid: %w", err)
	}
	if skew < 0 {
		return fmt.Errorf(
			"global.router.agentic_facts.bounds.clock_skew must be >= 0, got %q",
			bounds.ClockSkew,
		)
	}
	return nil
}
