package backendegress_test

import (
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendegress"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog"
)

func TestDefaultEgressPolicyCoversEveryFixedProviderOrigin(t *testing.T) {
	repositoryRoot := filepath.Clean(filepath.Join("..", "..", "..", ".."))
	policy, err := backendegress.LoadFile(filepath.Join(repositoryRoot, "config", "backend-egress-policy.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	for _, integration := range providercatalog.BuiltinIntegrations() {
		provider := integration.Definition()
		if provider.Origin.Mode != providercatalog.OriginFixed {
			continue
		}
		if _, err := policy.AuthorizeOrigin(provider.Origin.DefaultURL); err != nil {
			t.Fatalf("application Provider %q origin %q is not allowed: %v", provider.ID, provider.Origin.DefaultURL, err)
		}
	}
}
