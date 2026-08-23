package application

import (
	"testing"

	identitypostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementidentity/postgres"
)

func TestOptionalRecoveryServicePreservesDisabledState(t *testing.T) {
	var disabled *identitypostgres.RecoveryService
	if service := optionalRecoveryService(disabled); service != nil {
		t.Fatal("disabled recovery became a non-nil route service")
	}

	enabled := &identitypostgres.RecoveryService{}
	if service := optionalRecoveryService(enabled); service == nil {
		t.Fatal("configured recovery was omitted from route composition")
	}
}
