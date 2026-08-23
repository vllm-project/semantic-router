package quota

import (
	"errors"
	"fmt"
	"strconv"
	"strings"
)

var ErrInvalidCounterIdentity = errors.New("invalid quota counter identity")

// CounterKind is the runtime counter family selected by a rule's metric.
type CounterKind string

const (
	CounterKindRequests    CounterKind = "requests"
	CounterKindTokens      CounterKind = "tokens"
	CounterKindCost        CounterKind = "cost"
	CounterKindConcurrency CounterKind = "concurrency"
)

// CounterIdentity makes counter ownership explicit. A policy ID is
// intentionally absent: reusing a policy never shares state.
type CounterIdentity struct {
	BindingID string `json:"bindingId"`
	RuleID    string `json:"ruleId"`
}

func NewCounterIdentity(bindingID, ruleID string) (CounterIdentity, error) {
	identity := CounterIdentity{BindingID: bindingID, RuleID: ruleID}
	if err := identity.Validate(); err != nil {
		return CounterIdentity{}, err
	}
	return identity, nil
}

func (i CounterIdentity) Validate() error {
	if err := validateOpaqueID("binding ID", i.BindingID); err != nil {
		return fmt.Errorf("%w: %w", ErrInvalidCounterIdentity, err)
	}
	if err := validateOpaqueID("rule ID", i.RuleID); err != nil {
		return fmt.Errorf("%w: %w", ErrInvalidCounterIdentity, err)
	}
	return nil
}

// String returns an unambiguous length-prefixed representation. Storage
// adapters remain responsible for namespacing and escaping their own keys.
func (i CounterIdentity) String() string {
	return strconv.Itoa(len(i.BindingID)) + ":" + i.BindingID +
		strconv.Itoa(len(i.RuleID)) + ":" + i.RuleID
}

// CounterKind returns the counter family for a valid metric.
func (m Metric) CounterKind() (CounterKind, error) {
	switch m {
	case MetricRequests:
		return CounterKindRequests, nil
	case MetricInputTokens,
		MetricOutputTokens,
		MetricTotalTokens,
		MetricServedInputTokens,
		MetricServedOutputTokens,
		MetricServedTotalTokens:
		return CounterKindTokens, nil
	case MetricCost:
		return CounterKindCost, nil
	case MetricConcurrentRequests:
		return CounterKindConcurrency, nil
	default:
		return "", fmt.Errorf("%w: unsupported metric %q", ErrInvalidCounterIdentity, m)
	}
}

func validateOpaqueID(label, value string) error {
	if value == "" {
		return fmt.Errorf("%s is required", label)
	}
	if strings.TrimSpace(value) != value {
		return fmt.Errorf("%s must not contain surrounding whitespace", label)
	}
	if strings.ContainsRune(value, '\x00') {
		return fmt.Errorf("%s must not contain NUL", label)
	}
	return nil
}
