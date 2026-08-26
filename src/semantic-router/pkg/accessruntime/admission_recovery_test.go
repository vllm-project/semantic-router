package accessruntime

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotaruntime"
)

func TestAuthoritativeAdmissionRecoveryUsesExactDurableResourceID(t *testing.T) {
	tenant := TenantContext{
		NamespaceID: "namespace-1", APIKeyID: "key-1", UserID: "user-1", TeamID: "team-1",
		PolicyRevision: 7, BillingCurrency: "USD",
	}
	base := quotaruntime.AdmissionRecoveryContext{
		NamespaceID: tenant.NamespaceID,
		Principal: quotaruntime.RecoveryPrincipal{
			APIKeyID: tenant.APIKeyID, UserID: tenant.UserID, TeamID: tenant.TeamID,
		},
		Routing: quotaruntime.RecoveryRouting{
			EntrypointID: "ep_public_chat", AccessRevision: int64(tenant.PolicyRevision),
		},
		FallbackDispatch: quotaruntime.RecoveryDispatch{
			ModelID: "mdl_frontier", ModelRevision: 1, Currency: tenant.BillingCurrency,
		},
	}

	tests := []struct {
		name   string
		target Target
	}{
		{
			name: "entrypoint",
			target: Target{ResourceType: accesscontrol.GrantResourceEntrypoint,
				ResourceID: "ep_public_chat", Permission: accesscontrol.GrantPermissionInvoke},
		},
		{
			name: "model",
			target: Target{ResourceType: accesscontrol.GrantResourceModel,
				ResourceID: "mdl_frontier", Permission: accesscontrol.GrantPermissionInvoke},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			recovery, err := authoritativeAdmissionRecovery(tenant, test.target, &base)
			if err != nil {
				t.Fatalf("authoritativeAdmissionRecovery() error = %v", err)
			}
			if recovery == nil || recovery.Routing.EntrypointID != base.Routing.EntrypointID ||
				recovery.FallbackDispatch.ModelID != base.FallbackDispatch.ModelID {
				t.Fatalf("recovery resource identity changed: %+v", recovery)
			}
		})
	}
}

func TestAuthoritativeAdmissionRecoveryRejectsDifferentDurableResourceID(t *testing.T) {
	tenant := TenantContext{
		NamespaceID: "namespace-1", APIKeyID: "key-1", PolicyRevision: 7, BillingCurrency: "USD",
	}
	recovery := quotaruntime.AdmissionRecoveryContext{
		NamespaceID: tenant.NamespaceID,
		Principal:   quotaruntime.RecoveryPrincipal{APIKeyID: tenant.APIKeyID},
		Routing: quotaruntime.RecoveryRouting{
			EntrypointID: "ep_other", AccessRevision: int64(tenant.PolicyRevision),
		},
		FallbackDispatch: quotaruntime.RecoveryDispatch{Currency: tenant.BillingCurrency},
	}
	target := Target{ResourceType: accesscontrol.GrantResourceEntrypoint,
		ResourceID: "ep_public_chat", Permission: accesscontrol.GrantPermissionInvoke}
	if _, err := authoritativeAdmissionRecovery(tenant, target, &recovery); err == nil {
		t.Fatal("authoritativeAdmissionRecovery() accepted a different entrypoint identity")
	}
}
