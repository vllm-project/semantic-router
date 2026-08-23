package quotaruntime

import (
	"errors"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quota"
)

func TestCounterCorrectionSeparatesDebitFromAuthoritativeUsage(t *testing.T) {
	base := CounterCorrection{
		BindingID: "binding-1", RuleID: "rule-1", Metric: quota.MetricTotalTokens,
		Algorithm: quota.AlgorithmSlidingLog, Enforcement: quota.EnforcementEnforce,
		Amount: "100", CounterIncompleteCount: "2", ChargeAt: time.Now().UTC(),
		Window: time.Minute, Charge: true,
	}
	if _, _, err := validateCounterCorrection(base); err != nil {
		t.Fatalf("conservative debit validation: %v", err)
	}
	actual := base
	actual.Known = true
	if _, _, err := validateCounterCorrection(actual); err != nil {
		t.Fatalf("actual correction validation: %v", err)
	}
	waiver := base
	waiver.Amount, waiver.Charge, waiver.Known = "0", false, true
	if _, _, err := validateCounterCorrection(waiver); !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("known waiver error = %v, want %v", err, ErrInvalidRequest)
	}
}
