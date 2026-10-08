package helpers

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// WaitForActivatedResources requires Ready for the current generation of every resource.
func WaitForActivatedResources(ctx context.Context, interval time.Duration, resources []string, fetch func(context.Context, string) ([]byte, error)) error {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		ready := true
		for _, resource := range resources {
			data, err := fetch(ctx, resource)
			if err != nil || !currentGenerationReady(data) {
				ready = false
			}
		}
		if ready {
			return nil
		}
		select {
		case <-ctx.Done():
			return fmt.Errorf("waiting for CR runtime activation: %w", ctx.Err())
		case <-ticker.C:
		}
	}
}

func currentGenerationReady(data []byte) bool {
	var resource struct {
		Metadata struct {
			Generation int64 `json:"generation"`
		} `json:"metadata"`
		Status struct {
			Conditions []metav1.Condition `json:"conditions"`
		} `json:"status"`
	}
	if json.Unmarshal(data, &resource) != nil || resource.Metadata.Generation < 1 {
		return false
	}
	for _, condition := range resource.Status.Conditions {
		if condition.Type == "Ready" {
			return condition.Status == metav1.ConditionTrue && condition.ObservedGeneration == resource.Metadata.Generation
		}
	}
	return false
}
