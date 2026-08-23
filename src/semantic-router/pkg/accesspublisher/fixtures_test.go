package accesspublisher

import (
	"bytes"
	"encoding/json"
	"fmt"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessprojection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/providercredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

const (
	providerFixtureNamespaceID  = "11111111-1111-4111-8111-111111111111"
	providerFixtureCredentialID = "22222222-2222-4222-8222-222222222222"
	providerFixtureVersionID    = "33333333-3333-4333-8333-333333333333"
)

var fixtureTime = time.Date(2026, 8, 22, 2, 3, 4, 0, time.UTC)

func validDesiredState(revision uint64, requestLimit string) DesiredState {
	namespace := accesscontrol.Namespace{
		ID: "ns-publisher", Name: "Publisher", QuotaPartitionID: "partition-publisher", BillingCurrency: "USD",
		Status: accesscontrol.NamespaceStatusActive, Revision: accesscontrol.Revision(revision), RuntimeEpoch: 9,
		CreatedAt: fixtureTime, UpdatedAt: fixtureTime,
	}
	user := accesscontrol.User{
		NamespaceID: namespace.ID, ID: "user-publisher", Email: "publisher@example.com", DisplayName: "Publisher",
		Status: accesscontrol.UserStatusActive, CreatedAt: fixtureTime, UpdatedAt: fixtureTime,
	}
	team := accesscontrol.Team{
		NamespaceID: namespace.ID, ID: "team-publisher", Name: "Publisher team",
		Status: accesscontrol.TeamStatusActive, CreatedAt: fixtureTime, UpdatedAt: fixtureTime,
	}
	membership := accesscontrol.TeamMembership{
		NamespaceID: namespace.ID, TeamID: team.ID, UserID: user.ID, Role: accesscontrol.TeamRoleMember,
		Status: accesscontrol.MembershipStatusActive, CreatedAt: fixtureTime, UpdatedAt: fixtureTime,
	}
	key := accesscontrol.APIKey{
		NamespaceID: namespace.ID, ID: "key-publisher", Name: "Publisher key", Owner: user.SubjectRef(),
		ContextTeamID: team.ID, Status: accesscontrol.APIKeyStatusActive, PolicyEpoch: 1,
		DelegationEpoch: 1, Revision: accesscontrol.Revision(revision), CreatedAt: fixtureTime, UpdatedAt: fixtureTime,
	}
	accessPolicy := accesscontrol.AccessPolicy{
		NamespaceID: namespace.ID, ID: "access-publisher", DisplayName: "Publisher access",
		Status: accesscontrol.PolicyStatusActive, Revision: accesscontrol.Revision(revision),
		CreatedAt: fixtureTime, UpdatedAt: fixtureTime,
		Grants: []accesscontrol.AccessPolicyGrant{
			{PolicyID: "access-publisher", Resource: accesscontrol.GrantResource{Type: accesscontrol.GrantResourceEntrypoint, ID: "ep-chat"}, Permission: accesscontrol.GrantPermissionDiscover, Effect: accesscontrol.GrantEffectAllow},
			{PolicyID: "access-publisher", Resource: accesscontrol.GrantResource{Type: accesscontrol.GrantResourceEntrypoint, ID: "ep-chat"}, Permission: accesscontrol.GrantPermissionInvoke, Effect: accesscontrol.GrantEffectAllow},
		},
	}
	ratePolicy := accesscontrol.RateLimitPolicy{
		NamespaceID: namespace.ID, ID: "rate-publisher", DisplayName: "Publisher rate",
		Status: accesscontrol.PolicyStatusActive, Revision: accesscontrol.Revision(revision),
		CreatedAt: fixtureTime, UpdatedAt: fixtureTime,
		Rules: []accesscontrol.RateLimitRule{{
			ID: "rule-rpm", PolicyID: "rate-publisher", Metric: accesscontrol.RateMetricRequests,
			Algorithm: accesscontrol.RateAlgorithmSlidingLog, Limit: accesscontrol.QuotaValue(requestLimit),
			Window: time.Minute, Accounting: accesscontrol.RateAccountingRequest,
			Enforcement: accesscontrol.RateEnforcementEnforce,
		}},
	}
	candidate := accessprojection.Candidate{
		Revision: revision, Namespace: namespace, Key: key,
		Relationships: accesscontrol.APIKeyRelationships{OwnerUser: &user, ContextTeam: &team, ContextMembership: &membership},
		UserAccessBindings: []accesscontrol.AccessPolicyBinding{{
			ID: "access-binding-publisher", NamespaceID: namespace.ID, Subject: user.SubjectRef(), PolicyID: accessPolicy.ID,
			Status: accesscontrol.BindingStatusActive, Revision: accesscontrol.Revision(revision),
		}},
		AccessPolicies: map[accesscontrol.AccessPolicyID]accesscontrol.AccessPolicy{accessPolicy.ID: accessPolicy},
		TeamRateBindings: []accesscontrol.RateLimitBinding{{
			ID: "rate-binding-publisher", NamespaceID: namespace.ID, Subject: team.SubjectRef(), PolicyID: ratePolicy.ID,
			Mode: accesscontrol.RateBindingAllocation, QuotaPartitionID: namespace.QuotaPartitionID,
			Status: accesscontrol.BindingStatusActive, Revision: accesscontrol.Revision(revision),
		}},
		RatePolicies:  map[accesscontrol.RateLimitPolicyID]accesscontrol.RateLimitPolicy{ratePolicy.ID: ratePolicy},
		RoutingClaims: map[string]routingsnapshot.ClaimValue{"tier": {Kind: "string", String: "free"}},
	}
	inputPrice, outputPrice := "0.25", "1.00"
	routing := routingsnapshot.Bundle{
		NamespaceID: string(namespace.ID), Revision: int64(revision), Currency: "USD",
		Models: []routingsnapshot.Model{{
			ID: "model-chat", Revision: 1,
			CatalogRevision: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
			Name:            "local/chat", Capabilities: []string{"text"},
			Execution: routingsnapshot.ModelExecution{MaxRetries: 2, RequestTimeout: "30s", StreamTimeout: "60s"},
			Pricing:   routingsnapshot.ModelPricing{InputCostPerMillionTokens: &inputPrice, OutputCostPerMillionTokens: &outputPrice},
			Backends: []routingsnapshot.Backend{{
				ID: "backend-chat", ProviderID: "openai-compatible", WireFormat: "openai.chat.v1",
				Origin:          "https://models.example/v1",
				ProviderModelID: "chat", Connection: routingsnapshot.BackendConnection{Path: "/chat/completions"}, Weight: "1",
			}},
		}},
		Recipes: []routingsnapshot.Recipe{{
			ID: "recipe-chat", Revision: 1, Name: "Chat", Decisions: []routingsnapshot.Decision{{ID: "decision-chat", Name: "Chat", DispatchCardinality: routingsnapshot.DispatchCardinalitySingle}},
			Document: json.RawMessage(`{"signals":[],"decisions":[]}`),
		}},
		Entrypoints: []routingsnapshot.Entrypoint{{
			ID: "ep-chat", Revision: 1, Name: "Chat", Aliases: []string{"vllm-sr/chat"},
			Rules: []routingsnapshot.EntrypointRule{{
				ID: "rule-chat", Name: "Chat", RecipeID: "recipe-chat", RecipeRevision: 1,
				Assignments: map[string]routingsnapshot.AssignmentSet{
					"decision-chat": {Models: []routingsnapshot.Assignment{{ModelID: "model-chat", ModelRevision: 1, Weight: "1"}}},
				},
			}},
		}},
	}
	credential := accesscontrol.CredentialVersion{
		ID: accesscontrol.CredentialVersionID(fmt.Sprintf("credential-%d", revision)), APIKeyID: key.ID,
		KID: "publisherkid0001", SecretHMAC: bytes.Repeat([]byte{0x5a}, 32), PepperVersion: "pepper-1",
		Status: accesscontrol.CredentialStatusActive, NotBefore: fixtureTime, CreatedAt: fixtureTime,
	}
	return DesiredState{
		Namespace: namespace, Revision: revision, RevisionTime: fixtureTime.Add(time.Duration(revision) * time.Millisecond),
		Keys:        []accessprojection.Candidate{candidate},
		Credentials: []CredentialCandidate{{Kind: CredentialKindAPIKey, Credential: credential}}, Routing: routing,
	}
}

func mustPublication(t testing.TB, revision uint64, limit string) Publication {
	t.Helper()
	publication, err := Compile(validDesiredState(revision, limit))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	return publication
}

func desiredStateWithProviderCredential(t testing.TB, revision uint64) (DesiredState, providercredential.Codec) {
	t.Helper()
	state := validDesiredState(revision, "100")
	state.Namespace.ID = accesscontrol.NamespaceID(providerFixtureNamespaceID)
	state.Namespace.Revision = accesscontrol.Revision(revision)
	state.Keys = nil
	state.Credentials = nil
	state.Routing.NamespaceID = providerFixtureNamespaceID
	state.Routing.Revision = int64(revision)
	backend := &state.Routing.Models[0].Backends[0]
	backend.ProviderID = "openai"
	backend.Origin = "https://api.example.com/v1"
	backend.ProviderCredentialID = providerFixtureCredentialID
	activeVersion := providerFixtureVersionID
	credential := providercredential.Credential{
		ID: providerFixtureCredentialID, NamespaceID: providerFixtureNamespaceID, Name: "Primary",
		ProviderID: "openai", CredentialMode: providercredential.ModeRequired,
		CredentialAdapterID: "bearer",
		CatalogRevision:     "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		NormalizedOrigin:    "https://api.example.com/v1", Status: providercredential.StatusActive,
		ActiveVersionID: &activeVersion, Revision: revision, CreatedAt: fixtureTime, UpdatedAt: fixtureTime,
	}
	codec := providercredential.Codec{Keyring: accesscredential.KEKKeyring{
		ActiveVersion: "provider-kek-v1",
		Keys:          map[string][]byte{"provider-kek-v1": []byte("12345678901234567890123456789012")},
	}}
	version, err := codec.Seal(credential, activeVersion, []byte("provider-secret"), fixtureTime)
	if err != nil {
		t.Fatalf("seal provider credential fixture: %v", err)
	}
	state.ProviderCredentials = []ProviderCredentialCandidate{{
		Credential: credential, Versions: []providercredential.Version{version},
	}}
	return state, codec
}
