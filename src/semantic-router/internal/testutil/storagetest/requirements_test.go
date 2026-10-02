package storagetest

import (
	"context"
	"fmt"
	"math"
	"testing"
)

type recordingTest struct{ failure, skip string }

func (*recordingTest) Helper()                             {}
func (t *recordingTest) Fatalf(format string, args ...any) { t.failure = fmt.Sprintf(format, args...) }
func (t *recordingTest) Skipf(format string, args ...any)  { t.skip = fmt.Sprintf(format, args...) }

func TestRequiredStorageCannotSkipDependency(t *testing.T) {
	for _, enabled := range []string{"", "true", "false"} {
		t.Run(enabled, func(t *testing.T) {
			t.Setenv("VLLM_SR_REQUIRE_STORAGE_TESTS", "1")
			t.Setenv("SKIP_REDIS_TESTS", enabled)
			result := &recordingTest{}
			Require(result, "redis")
			if enabled == "false" {
				if result.failure != "" || result.skip != "" {
					t.Fatal(result)
				}
				Unavailable(result, "redis", "connection refused")
			}
			if result.failure == "" || result.skip != "" {
				t.Fatalf("mandatory dependency silently skipped: %+v", result)
			}
		})
	}
}

func TestOptionalStorageRequiresExplicitOptIn(t *testing.T) {
	t.Setenv("VLLM_SR_REQUIRE_STORAGE_TESTS", "")
	t.Setenv("SKIP_REDIS_TESTS", "")
	result := &recordingTest{}
	Require(result, "redis")
	if result.skip == "" || result.failure != "" {
		t.Fatal(result)
	}
}

func TestStorageVectorsHaveStableGeometryAndCancellation(t *testing.T) {
	provider := Vectors{Size: 384, Aliases: map[string]string{"query": "record"}}
	a, _ := provider.Embed(context.Background(), "record")
	b, _ := provider.Embed(context.Background(), "query")
	c, _ := provider.Embed(context.Background(), "unrelated")
	var same, different float64
	for i := range a {
		same += float64(a[i] * b[i])
		different += float64(a[i] * c[i])
	}
	if math.Abs(same-1) > 1e-5 || math.Abs(different) > .3 {
		t.Fatalf("fixture cosine same=%f unrelated=%f", same, different)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := provider.Embed(ctx, "record"); err == nil {
		t.Fatal("cancelled call succeeded")
	}
}
