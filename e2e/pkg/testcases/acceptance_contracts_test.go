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
		"domain-classify",
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

	if !strings.Contains(err.Error(), "below minimum 60.00%") {
		t.Fatalf("unexpected error: %v", err)
	}

	if got := details["minimum_accuracy_rate"]; got != "60.00%" {
		t.Fatalf("minimum_accuracy_rate = %v, want 60.00%%", got)
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

func TestWrapWithAcceptanceContractAddsProgressiveOverallDetails(t *testing.T) {
	wrapped := wrapWithAcceptanceContract(
		"chat-completions-progressive-stress",
		func(ctx context.Context, client *kubernetes.Clientset, opts TestCaseOptions) error {
			opts.SetDetails(map[string]interface{}{
				"qps_10": map[string]interface{}{
					"total_requests": 10,
					"successful":     10,
					"failed":         0,
					"success_rate":   "100.00%",
				},
				"qps_20": map[string]interface{}{
					"total_requests": 20,
					"successful":     18,
					"failed":         2,
					"success_rate":   "90.00%",
				},
				"qps_50": map[string]interface{}{
					"total_requests": 50,
					"successful":     40,
					"failed":         10,
					"success_rate":   "80.00%",
				},
			})
			return nil
		},
	)

	var details map[string]interface{}
	if err := wrapped(context.Background(), nil, TestCaseOptions{
		Profile: baselineRouterContractProfile,
		SetDetails: func(next map[string]interface{}) {
			details = next
		},
	}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	overall, ok := details["overall"].(map[string]interface{})
	if !ok {
		t.Fatalf("overall details missing or wrong type: %#v", details["overall"])
	}

	if got := overall["minimum_success_rate"]; got != "75.00%" {
		t.Fatalf("overall minimum_success_rate = %v, want 75.00%%", got)
	}
}

func TestWrapWithAcceptanceContractPreservesOriginalError(t *testing.T) {
	sentinel := errors.New("transport failure")
	wrapped := wrapWithAcceptanceContract(
		"domain-classify",
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
