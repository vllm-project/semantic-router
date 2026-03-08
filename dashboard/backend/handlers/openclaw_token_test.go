package handlers

import "testing"

func TestEmbeddedGatewayToken_HidesRawToken(t *testing.T) {
	h := NewOpenClawHandler(t.TempDir(), false)
	workerName := "worker-c"

	if err := h.saveRegistry([]ContainerEntry{
		{
			Name:  workerName,
			Token: "raw-secret-token",
		},
	}); err != nil {
		t.Fatalf("failed to seed registry: %v", err)
	}

	if got := h.embeddedGatewayToken(workerName); got == "" || got == "raw-secret-token" {
		t.Fatalf("expected mediated embedded token, got %q", got)
	}
}
