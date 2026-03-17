package helm

import "testing"

func TestInstallTimeoutForRelease(t *testing.T) {
	t.Run("non semantic router release ignores override", func(t *testing.T) {
		t.Setenv(semanticRouterHelmTimeoutEnv, "45m")

		timeout, err := installTimeoutForRelease("envoy-gateway", "10m")
		if err != nil {
			t.Fatalf("installTimeoutForRelease returned error: %v", err)
		}
		if timeout != "10m" {
			t.Fatalf("expected existing timeout to be preserved, got %q", timeout)
		}
	})

	t.Run("semantic router uses override when larger", func(t *testing.T) {
		t.Setenv(semanticRouterHelmTimeoutEnv, "45m")

		timeout, err := installTimeoutForRelease("semantic-router", "30m")
		if err != nil {
			t.Fatalf("installTimeoutForRelease returned error: %v", err)
		}
		if timeout != "45m" {
			t.Fatalf("expected override timeout, got %q", timeout)
		}
	})

	t.Run("semantic router preserves longer configured timeout", func(t *testing.T) {
		t.Setenv(semanticRouterHelmTimeoutEnv, "45m")

		timeout, err := installTimeoutForRelease("semantic-router", "60m")
		if err != nil {
			t.Fatalf("installTimeoutForRelease returned error: %v", err)
		}
		if timeout != "60m" {
			t.Fatalf("expected longer configured timeout, got %q", timeout)
		}
	})

	t.Run("semantic router uses override when timeout is unset", func(t *testing.T) {
		t.Setenv(semanticRouterHelmTimeoutEnv, "45m")

		timeout, err := installTimeoutForRelease("semantic-router", "")
		if err != nil {
			t.Fatalf("installTimeoutForRelease returned error: %v", err)
		}
		if timeout != "45m" {
			t.Fatalf("expected override timeout, got %q", timeout)
		}
	})

	t.Run("invalid override returns error", func(t *testing.T) {
		t.Setenv(semanticRouterHelmTimeoutEnv, "not-a-duration")

		if _, err := installTimeoutForRelease("semantic-router", "30m"); err == nil {
			t.Fatal("expected invalid duration error")
		}
	})
}
