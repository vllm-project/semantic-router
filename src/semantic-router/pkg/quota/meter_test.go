package quota

import (
	"encoding/json"
	"errors"
	"testing"
)

func TestNewPublicMeterCompleteStates(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name          string
		used          string
		wantRemaining string
		wantOverage   string
		wantState     CapacityState
	}{
		{name: "available", used: "2", wantRemaining: "10", wantState: CapacityAvailable},
		{name: "exhausted", used: "12", wantRemaining: "0", wantState: CapacityExhausted},
		{name: "over limit", used: "15", wantRemaining: "0", wantOverage: "3", wantState: CapacityOverLimit},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			meter, err := NewPublicMeter(integerMeterSnapshot(t, "12", test.used, "18", "0"))
			if err != nil {
				t.Fatalf("NewPublicMeter() error = %v", err)
			}
			if meter.Completeness != CompletenessComplete {
				t.Errorf("Completeness = %q, want %q", meter.Completeness, CompletenessComplete)
			}
			if meter.CapacityState != test.wantState {
				t.Errorf("CapacityState = %q, want %q", meter.CapacityState, test.wantState)
			}
			if meter.Remaining == nil || *meter.Remaining != test.wantRemaining {
				t.Errorf("Remaining = %v, want %q", meter.Remaining, test.wantRemaining)
			}
			if test.wantOverage == "" && meter.Overage != nil {
				t.Errorf("Overage = %v, want nil", meter.Overage)
			}
			if test.wantOverage != "" && (meter.Overage == nil || *meter.Overage != test.wantOverage) {
				t.Errorf("Overage = %v, want %q", meter.Overage, test.wantOverage)
			}
		})
	}
}

func TestNewPublicMeterIncompleteStates(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name             string
		enforcement      Enforcement
		known            string
		incomplete       string
		fenceOpen        bool
		wantCompleteness Completeness
		wantCapacity     CapacityState
	}{
		{
			name: "enforced partial is fenced", enforcement: EnforcementEnforce,
			known: "2", incomplete: "1", fenceOpen: true,
			wantCompleteness: CompletenessPartial, wantCapacity: CapacityFenced,
		},
		{
			name: "enforced unknown is fenced", enforcement: EnforcementEnforce,
			known: "0", incomplete: "1",
			wantCompleteness: CompletenessUnknown, wantCapacity: CapacityFenced,
		},
		{
			name: "corrected fence stays closed through ledger commit", enforcement: EnforcementEnforce,
			known: "3", incomplete: "0", fenceOpen: true,
			wantCompleteness: CompletenessComplete, wantCapacity: CapacityFenced,
		},
		{
			name: "shadow partial is unknown", enforcement: EnforcementShadow,
			known: "2", incomplete: "1",
			wantCompleteness: CompletenessPartial, wantCapacity: CapacityUnknown,
		},
		{
			name: "shadow unknown is unknown", enforcement: EnforcementShadow,
			known: "0", incomplete: "1",
			wantCompleteness: CompletenessUnknown, wantCapacity: CapacityUnknown,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			snapshot := integerMeterSnapshot(t, "12", "2", test.known, test.incomplete)
			snapshot.Enforcement = test.enforcement
			snapshot.FenceOpen = test.fenceOpen
			meter, err := NewPublicMeter(snapshot)
			if err != nil {
				t.Fatalf("NewPublicMeter() error = %v", err)
			}
			if meter.Completeness != test.wantCompleteness {
				t.Errorf("Completeness = %q, want %q", meter.Completeness, test.wantCompleteness)
			}
			if meter.CapacityState != test.wantCapacity {
				t.Errorf("CapacityState = %q, want %q", meter.CapacityState, test.wantCapacity)
			}
			if meter.Remaining != nil {
				t.Errorf("Remaining = %v, want nil", meter.Remaining)
			}
		})
	}
}

func TestNewPublicMeterEmptyIsCompleteZero(t *testing.T) {
	t.Parallel()

	meter, err := NewPublicMeter(integerMeterSnapshot(t, "12", "0", "0", "0"))
	if err != nil {
		t.Fatalf("NewPublicMeter() error = %v", err)
	}
	if meter.Completeness != CompletenessComplete || meter.CapacityState != CapacityAvailable {
		t.Fatalf("empty state = (%q, %q), want (%q, %q)", meter.Completeness, meter.CapacityState, CompletenessComplete, CapacityAvailable)
	}
	if meter.Remaining == nil || *meter.Remaining != "12" {
		t.Fatalf("Remaining = %v, want 12", meter.Remaining)
	}
}

func TestNewPublicMeterCostUsesCanonicalCurrencyDecimals(t *testing.T) {
	t.Parallel()

	counter, err := NewCounterIdentity("binding-cost", "rule-cost")
	if err != nil {
		t.Fatalf("NewCounterIdentity() error = %v", err)
	}
	snapshot := MeterSnapshot{
		Counter: counter, Metric: MetricCost, Enforcement: EnforcementEnforce,
		Limit:    mustCurrencyDecimal(t, "5").ScaledInteger(),
		Used:     mustCurrencyDecimal(t, "2.5").ScaledInteger(),
		Currency: "USD", KnownDispatches: mustQuotaInteger(t, "18"),
	}
	meter, err := NewPublicMeter(snapshot)
	if err != nil {
		t.Fatalf("NewPublicMeter() error = %v", err)
	}
	if meter.Limit != "5" || meter.Used != "2.5" || meter.Remaining == nil || *meter.Remaining != "2.5" {
		t.Fatalf("cost values = (%q, %q, %v), want (5, 2.5, 2.5)", meter.Limit, meter.Used, meter.Remaining)
	}
	if meter.Currency != "USD" {
		t.Fatalf("Currency = %q, want USD", meter.Currency)
	}
}

func TestNewPublicMeterRejectsInconsistentSnapshots(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		mutate func(*MeterSnapshot)
	}{
		{name: "invalid counter", mutate: func(snapshot *MeterSnapshot) { snapshot.Counter.BindingID = "" }},
		{name: "invalid metric", mutate: func(snapshot *MeterSnapshot) { snapshot.Metric = "bogus" }},
		{name: "invalid enforcement", mutate: func(snapshot *MeterSnapshot) { snapshot.Enforcement = "bogus" }},
		{name: "zero limit", mutate: func(snapshot *MeterSnapshot) { snapshot.Limit = QuotaInteger{} }},
		{name: "currency on requests", mutate: func(snapshot *MeterSnapshot) { snapshot.Currency = "USD" }},
		{name: "shadow fence", mutate: func(snapshot *MeterSnapshot) {
			snapshot.Enforcement = EnforcementShadow
			snapshot.IncompleteDispatches = mustQuotaInteger(t, "1")
			snapshot.FenceOpen = true
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			snapshot := integerMeterSnapshot(t, "12", "2", "1", "0")
			test.mutate(&snapshot)
			if _, err := NewPublicMeter(snapshot); !errors.Is(err, ErrInvalidMeter) {
				t.Fatalf("NewPublicMeter() error = %v, want %v", err, ErrInvalidMeter)
			}
		})
	}

	cost := integerMeterSnapshot(t, "12", "2", "1", "0")
	cost.Metric = MetricCost
	if _, err := NewPublicMeter(cost); !errors.Is(err, ErrInvalidMeter) {
		t.Fatalf("cost without currency error = %v, want %v", err, ErrInvalidMeter)
	}
	cost.Currency = "usd"
	if _, err := NewPublicMeter(cost); !errors.Is(err, ErrInvalidMeter) {
		t.Fatalf("lowercase currency error = %v, want %v", err, ErrInvalidMeter)
	}
}

func TestPublicMeterKeepsCorrectedFenceClosedUntilLedgerSettlement(t *testing.T) {
	t.Parallel()

	snapshot := integerMeterSnapshot(t, "12", "2", "2", "0")
	snapshot.FenceOpen = true
	meter, err := NewPublicMeter(snapshot)
	if err != nil {
		t.Fatalf("NewPublicMeter() error = %v", err)
	}
	if meter.CapacityState != CapacityFenced || meter.Completeness != CompletenessComplete ||
		meter.Remaining != nil || meter.Overage != nil {
		t.Fatalf("corrected fenced meter = %#v, want complete usage with undisclosed fenced capacity", meter)
	}
}

func TestPublicMeterJSONKeepsUnknownRemainingNull(t *testing.T) {
	t.Parallel()

	snapshot := integerMeterSnapshot(t, "12", "2", "0", "1")
	meter, err := NewPublicMeter(snapshot)
	if err != nil {
		t.Fatalf("NewPublicMeter() error = %v", err)
	}
	encoded, err := json.Marshal(meter)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	var document map[string]any
	if err := json.Unmarshal(encoded, &document); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	if value, present := document["remaining"]; !present || value != nil {
		t.Fatalf("remaining JSON = (%v, %t), want explicit null", value, present)
	}
}

func integerMeterSnapshot(t *testing.T, limit, used, known, incomplete string) MeterSnapshot {
	t.Helper()
	counter, err := NewCounterIdentity("binding-1", "rule-1")
	if err != nil {
		t.Fatalf("NewCounterIdentity() error = %v", err)
	}
	return MeterSnapshot{
		Counter: counter, Metric: MetricRequests, Enforcement: EnforcementEnforce,
		Limit: mustQuotaInteger(t, limit), Used: mustQuotaInteger(t, used),
		KnownDispatches:      mustQuotaInteger(t, known),
		IncompleteDispatches: mustQuotaInteger(t, incomplete),
	}
}
