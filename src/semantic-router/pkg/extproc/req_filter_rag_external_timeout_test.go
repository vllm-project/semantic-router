package extproc

import (
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestExternalAPIClientHonorsConfiguredTimeout(t *testing.T) {
	sixty := 60
	cfg := &config.ExternalAPIRAGConfig{TimeoutSeconds: &sixty}
	if got := externalAPIClient(cfg.GetTimeout()).Timeout; got != 60*time.Second {
		t.Fatalf("client timeout = %v, want 60s", got)
	}
	if externalAPIClient(time.Second).Transport != externalAPIClient(time.Minute).Transport {
		t.Fatal("clients must share one transport")
	}
}
