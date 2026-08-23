package quota

import (
	"errors"
	"testing"
)

func TestCounterIdentityUsesBindingAndRule(t *testing.T) {
	t.Parallel()

	first, err := NewCounterIdentity("binding-a", "rule-a")
	if err != nil {
		t.Fatalf("NewCounterIdentity() error = %v", err)
	}
	second, err := NewCounterIdentity("binding-b", "rule-a")
	if err != nil {
		t.Fatalf("NewCounterIdentity() error = %v", err)
	}
	if first == second {
		t.Fatal("different bindings unexpectedly share a counter identity")
	}
	if first.String() == second.String() {
		t.Fatal("different bindings unexpectedly share a canonical identity")
	}

	ambiguousLeft, err := NewCounterIdentity("a:b", "c")
	if err != nil {
		t.Fatalf("NewCounterIdentity() error = %v", err)
	}
	ambiguousRight, err := NewCounterIdentity("a", "b:c")
	if err != nil {
		t.Fatalf("NewCounterIdentity() error = %v", err)
	}
	if ambiguousLeft.String() == ambiguousRight.String() {
		t.Fatal("length-prefixed identity is ambiguous")
	}
}

func TestCounterIdentityRejectsInvalidIDs(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		bindingID string
		ruleID    string
	}{
		{name: "missing binding", ruleID: "rule"},
		{name: "missing rule", bindingID: "binding"},
		{name: "binding whitespace", bindingID: " binding", ruleID: "rule"},
		{name: "rule whitespace", bindingID: "binding", ruleID: "rule 	"},
		{name: "binding NUL", bindingID: "binding\x00bad", ruleID: "rule"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			_, err := NewCounterIdentity(test.bindingID, test.ruleID)
			if !errors.Is(err, ErrInvalidCounterIdentity) {
				t.Fatalf("NewCounterIdentity() error = %v, want %v", err, ErrInvalidCounterIdentity)
			}
		})
	}
}

func TestMetricCounterKind(t *testing.T) {
	t.Parallel()

	tests := []struct {
		metric Metric
		want   CounterKind
	}{
		{metric: MetricRequests, want: CounterKindRequests},
		{metric: MetricInputTokens, want: CounterKindTokens},
		{metric: MetricOutputTokens, want: CounterKindTokens},
		{metric: MetricTotalTokens, want: CounterKindTokens},
		{metric: MetricServedInputTokens, want: CounterKindTokens},
		{metric: MetricServedOutputTokens, want: CounterKindTokens},
		{metric: MetricServedTotalTokens, want: CounterKindTokens},
		{metric: MetricCost, want: CounterKindCost},
		{metric: MetricConcurrentRequests, want: CounterKindConcurrency},
	}
	for _, test := range tests {
		got, err := test.metric.CounterKind()
		if err != nil {
			t.Errorf("CounterKind(%q) error = %v", test.metric, err)
			continue
		}
		if got != test.want {
			t.Errorf("CounterKind(%q) = %q, want %q", test.metric, got, test.want)
		}
	}
	if _, err := Metric("bogus").CounterKind(); !errors.Is(err, ErrInvalidCounterIdentity) {
		t.Fatalf("CounterKind(bogus) error = %v, want %v", err, ErrInvalidCounterIdentity)
	}
}
