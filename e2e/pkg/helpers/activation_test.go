package helpers

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"
)

func TestWaitForActivatedCRsRequiresCurrentGenerationOfBothResources(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	calls := 0
	err := WaitForActivatedResources(ctx, time.Millisecond, []string{"intelligentpool/ai-gateway-pool", "intelligentroute/ai-gateway-route"}, func(_ context.Context, resource string) ([]byte, error) {
		calls++
		if strings.HasPrefix(resource, "intelligentroute/") && calls < 4 {
			return []byte(`{"metadata":{"generation":2},"status":{"conditions":[{"type":"Ready","status":"True","observedGeneration":1}]}}`), nil
		}
		return []byte(`{"metadata":{"generation":2},"status":{"conditions":[{"type":"Ready","status":"True","observedGeneration":2}]}}`), nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if calls < 4 {
		t.Fatal("stale Ready condition was accepted")
	}
}

func TestWaitForActivatedCRsCancels(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	err := WaitForActivatedResources(ctx, time.Hour, []string{"intelligentpool/ai-gateway-pool", "intelligentroute/ai-gateway-route"}, func(context.Context, string) ([]byte, error) { return nil, errors.New("not ready") })
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("wait = %v", err)
	}
}
