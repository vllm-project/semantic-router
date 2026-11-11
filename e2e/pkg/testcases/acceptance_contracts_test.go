package testcases

import (
	"context"
	"errors"
	"strings"
	"testing"

	"k8s.io/client-go/kubernetes"
)

func TestWrapWithAcceptanceContractAppliesFlatThresholds(t *testing.T) {
	wrapped := wrapWithAcceptanceContract(
		"decision-priority-selection",
		func(ctx context.Context, client *kubernetes.Clientset, opts TestCaseOptions) error {
			opts.SetDetails(map[string]interface{}{
				"total_tests":   10,
				"correct_tests": 5,
				"accuracy_rate": "50.00%",
				"failed_tests":  5,
			})
			return nil
		},
	)

	var details map[string]interface{}
	err := wrapped(context.Background(), nil, TestCaseOptions{
		Profile: baselineRouterContractProfile,
		SetDetails: func(next map[string]interface{}) {
			details = next
		},
	})
	if err == nil {
		t.Fatal("expected acceptance failure")
	}

	if !strings.Contains(err.Error(), "below minimum 75.00%") {
		t.Fatalf("unexpected error: %v", err)
	}

	if got := details["minimum_accuracy_rate"]; got != "75.00%" {
		t.Fatalf("minimum_accuracy_rate = %v, want 75.00%%", got)
	}
}

func TestWrapWithAcceptanceContractSkipsUnscopedProfiles(t *testing.T) {
	wrapped := wrapWithAcceptanceContract(
		"decision-priority-selection",
		func(ctx context.Context, client *kubernetes.Clientset, opts TestCaseOptions) error {
			opts.SetDetails(map[string]interface{}{
				"total_tests":   4,
				"correct_tests": 1,
				"accuracy_rate": "25.00%",
				"failed_tests":  3,
			})
			return nil
		},
	)

	var details map[string]interface{}
	if err := wrapped(context.Background(), nil, TestCaseOptions{
		Profile: "dynamic-config",
		SetDetails: func(next map[string]interface{}) {
			details = next
		},
	}); err != nil {
		t.Fatalf("expected non-baseline profile to skip acceptance contract, got %v", err)
	}

	if got := details["minimum_accuracy_rate"]; got != nil {
		t.Fatalf("minimum_accuracy_rate = %v, want nil", got)
	}
}

func progressiveStressDetails(answered50 int) map[string]interface{} {
	stage := func(total, answered int) map[string]interface{} {
		return map[string]interface{}{
			"total_requests": total,
			"successful":     answered,
			"failed":         total - answered,
		}
	}
	return map[string]interface{}{
		"qps_10": stage(10, 10),
		"qps_20": stage(20, 20),
		"qps_50": stage(50, answered50),
	}
}

func runProgressiveStressContract(t *testing.T, answered50 int) (map[string]interface{}, error) {
	t.Helper()
	wrapped := wrapWithAcceptanceContract(
		"chat-completions-progressive-stress",
		func(ctx context.Context, client *kubernetes.Clientset, opts TestCaseOptions) error {
			opts.SetDetails(progressiveStressDetails(answered50))
			return nil
		},
	)

	var details map[string]interface{}
	err := wrapped(context.Background(), nil, TestCaseOptions{
		Profile: baselineRouterContractProfile,
		SetDetails: func(next map[string]interface{}) {
			details = next
		},
	})
	return details, err
}

func TestWrapWithAcceptanceContractAddsProgressiveOverallDetails(t *testing.T) {
	details, err := runProgressiveStressContract(t, 50)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	overall, ok := details["overall"].(map[string]interface{})
	if !ok {
		t.Fatalf("overall details missing or wrong type: %#v", details["overall"])
	}

	if got := overall["minimum_success_rate"]; got != "100.00%" {
		t.Fatalf("overall minimum_success_rate = %v, want 100.00%%", got)
	}
}

// The client waits out the gateway's deadline, so a missing answer is a
// failure of the gateway or the Router at any stage.
func TestProgressiveStressContractRejectsOneUnansweredRequest(t *testing.T) {
	_, err := runProgressiveStressContract(t, 49)
	if err == nil {
		t.Fatal("expected one unanswered request to fail the contract")
	}
	if !strings.Contains(err.Error(), "49/50 qps_50 successful requests") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestWrapWithAcceptanceContractPreservesOriginalError(t *testing.T) {
	sentinel := errors.New("transport failure")
	wrapped := wrapWithAcceptanceContract(
		"decision-priority-selection",
		func(ctx context.Context, client *kubernetes.Clientset, opts TestCaseOptions) error {
			opts.SetDetails(map[string]interface{}{
				"total_tests":   10,
				"correct_tests": 0,
				"accuracy_rate": "0.00%",
				"failed_tests":  10,
			})
			return sentinel
		},
	)

	err := wrapped(context.Background(), nil, TestCaseOptions{
		Profile:    baselineRouterContractProfile,
		SetDetails: func(next map[string]interface{}) {},
	})
	if !errors.Is(err, sentinel) {
		t.Fatalf("expected original error, got %v", err)
	}
}
