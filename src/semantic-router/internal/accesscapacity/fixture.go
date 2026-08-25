package accesscapacity

import (
	"encoding/json"
	"fmt"
	"strconv"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessprojection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesspublisher"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

const (
	fixtureNamespaceID = "capacity-namespace"
	fixturePartition   = "capacity-partition"
	fixturePepper      = "capacity-pepper-v1"
	fixtureModelID     = "capacity-model"
	fixtureRecipeID    = "capacity-recipe"
	fixtureDecisionID  = "capacity-decision"
	fixtureEntrypointA = "capacity-entrypoint-a"
	fixtureEntrypointB = "capacity-entrypoint-b"
)

type Fixture struct {
	Desired     accesspublisher.DesiredState
	Credentials []string
	Targets     []string
	Keyring     accesscredential.PepperKeyring
}

func BuildFixture(config Config, now time.Time) (Fixture, error) {
	now = now.UTC().Truncate(time.Millisecond)
	keyring := accesscredential.PepperKeyring{
		ActiveVersion: fixturePepper,
		Keys: map[string][]byte{
			fixturePepper: []byte("capacity-gate-ephemeral-pepper-material-v1"),
		},
	}
	if err := keyring.Validate(); err != nil {
		return Fixture{}, fmt.Errorf("build keyring: %w", err)
	}
	namespace, user, team, membership := fixtureIdentity(now)
	state := accesspublisher.DesiredState{
		Namespace: namespace, Revision: 1, RevisionTime: now,
		Keys:        make([]accessprojection.Candidate, 0, config.KeyCount),
		Credentials: make([]accesspublisher.CredentialCandidate, 0, config.KeyCount),
		Routing:     fixtureRoutingBundle(),
	}
	fixture := Fixture{
		Desired: state, Credentials: make([]string, 0, config.KeyCount),
		Targets: make([]string, 0, config.KeyCount), Keyring: keyring,
	}
	for index := range config.KeyCount {
		candidate, credential, target, err := fixtureKey(
			config, now, index, namespace, user, team, membership, keyring,
		)
		if err != nil {
			return Fixture{}, err
		}
		fixture.Desired.Keys = append(fixture.Desired.Keys, candidate)
		fixture.Desired.Credentials = append(fixture.Desired.Credentials, accesspublisher.CredentialCandidate{
			Kind: accesspublisher.CredentialKindAPIKey, Credential: accesscontrol.CredentialVersion{
				ID:       accesscontrol.CredentialVersionID("capacity-credential-" + suffix(index)),
				APIKeyID: candidate.Key.ID, KID: credential.Digest.PublicID,
				SecretHMAC:    append([]byte(nil), credential.Digest.HMAC...),
				PepperVersion: credential.Digest.PepperVersion,
				Status:        accesscontrol.CredentialStatusActive,
				NotBefore:     now.Add(-time.Minute), CreatedAt: now.Add(-time.Minute),
			},
		})
		fixture.Credentials = append(fixture.Credentials, credential.Plaintext)
		fixture.Targets = append(fixture.Targets, target)
	}
	return fixture, nil
}

func fixtureIdentity(now time.Time) (
	accesscontrol.Namespace,
	accesscontrol.User,
	accesscontrol.Team,
	accesscontrol.TeamMembership,
) {
	namespace := accesscontrol.Namespace{
		ID: fixtureNamespaceID, Name: "Capacity gate", QuotaPartitionID: fixturePartition,
		BillingCurrency: "USD", Status: accesscontrol.NamespaceStatusActive,
		Revision: 1, RuntimeEpoch: 1, CreatedAt: now, UpdatedAt: now,
	}
	user := accesscontrol.User{
		NamespaceID: namespace.ID, ID: "capacity-user", Email: "capacity@example.invalid",
		DisplayName: "Capacity gate", Status: accesscontrol.UserStatusActive,
		CreatedAt: now, UpdatedAt: now,
	}
	team := accesscontrol.Team{
		NamespaceID: namespace.ID, ID: "capacity-team", Name: "Capacity gate",
		Status: accesscontrol.TeamStatusActive, CreatedAt: now, UpdatedAt: now,
	}
	membership := accesscontrol.TeamMembership{
		NamespaceID: namespace.ID, TeamID: team.ID, UserID: user.ID,
		Role: accesscontrol.TeamRoleMember, Status: accesscontrol.MembershipStatusActive,
		CreatedAt: now, UpdatedAt: now,
	}
	return namespace, user, team, membership
}

func fixtureKey(
	config Config,
	now time.Time,
	index int,
	namespace accesscontrol.Namespace,
	user accesscontrol.User,
	team accesscontrol.Team,
	membership accesscontrol.TeamMembership,
	keyring accesscredential.PepperKeyring,
) (accessprojection.Candidate, accesscredential.Issued, string, error) {
	idSuffix := suffix(index)
	keyID := accesscontrol.APIKeyID("capacity-key-" + idSuffix)
	accessPolicyID := accesscontrol.AccessPolicyID("capacity-access-" + idSuffix)
	ratePolicyID := accesscontrol.RateLimitPolicyID("capacity-rate-" + idSuffix)
	target := fixtureEntrypointA
	if index%2 == 1 {
		target = fixtureEntrypointB
	}
	key := accesscontrol.APIKey{
		NamespaceID: namespace.ID, ID: keyID, Name: "Capacity key " + idSuffix,
		Owner: user.SubjectRef(), ContextTeamID: team.ID, Status: accesscontrol.APIKeyStatusActive,
		PolicyEpoch: 1, DelegationEpoch: 1, Revision: 1, CreatedAt: now, UpdatedAt: now,
	}
	accessPolicy := accesscontrol.AccessPolicy{
		NamespaceID: namespace.ID, ID: accessPolicyID, DisplayName: "Capacity access " + idSuffix,
		Status: accesscontrol.PolicyStatusActive, Revision: 1, CreatedAt: now, UpdatedAt: now,
		Grants: []accesscontrol.AccessPolicyGrant{{
			PolicyID: accessPolicyID,
			Resource: accesscontrol.GrantResource{
				Type: accesscontrol.GrantResourceEntrypoint, ID: accesscontrol.ResourceID(target),
			},
			Permission: accesscontrol.GrantPermissionInvoke, Effect: accesscontrol.GrantEffectAllow,
		}},
	}
	ratePolicy := accesscontrol.RateLimitPolicy{
		NamespaceID: namespace.ID, ID: ratePolicyID, DisplayName: "Capacity rate " + idSuffix,
		Status: accesscontrol.PolicyStatusActive, Revision: 1, CreatedAt: now, UpdatedAt: now,
		Rules: []accesscontrol.RateLimitRule{{
			ID: accesscontrol.RateLimitRuleID("capacity-rule-" + idSuffix), PolicyID: ratePolicyID,
			Metric: accesscontrol.RateMetricRequests, Algorithm: accesscontrol.RateAlgorithmSlidingLog,
			Limit: accesscontrol.QuotaValue(strconv.Itoa(config.RequestLimit)), Window: time.Hour,
			Accounting:  accesscontrol.RateAccountingRequest,
			Enforcement: accesscontrol.RateEnforcementEnforce,
		}},
	}
	candidate := accessprojection.Candidate{
		Revision: 1, Namespace: namespace, Key: key,
		Relationships: accesscontrol.APIKeyRelationships{
			OwnerUser: &user, ContextTeam: &team, ContextMembership: &membership,
		},
		KeyAccessBindings: []accesscontrol.AccessPolicyBinding{{
			ID:          accesscontrol.PolicyBindingID("capacity-access-binding-" + idSuffix),
			NamespaceID: namespace.ID, Subject: key.SubjectRef(), PolicyID: accessPolicyID,
			Status: accesscontrol.BindingStatusActive, Revision: 1,
		}},
		AccessPolicies: map[accesscontrol.AccessPolicyID]accesscontrol.AccessPolicy{
			accessPolicyID: accessPolicy,
		},
		KeyRateBindings: []accesscontrol.RateLimitBinding{{
			ID:          accesscontrol.PolicyBindingID("capacity-rate-binding-" + idSuffix),
			NamespaceID: namespace.ID, Subject: key.SubjectRef(), PolicyID: ratePolicyID,
			Mode: accesscontrol.RateBindingAllocation, QuotaPartitionID: namespace.QuotaPartitionID,
			Status: accesscontrol.BindingStatusActive, Revision: 1,
		}},
		RatePolicies: map[accesscontrol.RateLimitPolicyID]accesscontrol.RateLimitPolicy{
			ratePolicyID: ratePolicy,
		},
		RoutingClaims: map[string]routingsnapshot.ClaimValue{
			"capacity-key": {Kind: "string", String: idSuffix},
		},
	}
	issued, err := keyring.Issue(accesscredential.KindAPIKey, "capacitykid"+idSuffix)
	if err != nil {
		return accessprojection.Candidate{}, accesscredential.Issued{}, "", fmt.Errorf(
			"issue capacity credential %d: %w", index, err,
		)
	}
	return candidate, issued, target, nil
}

func fixtureRoutingBundle() routingsnapshot.Bundle {
	inputPrice, outputPrice := "0.1", "0.2"
	entrypoint := func(id, alias string) routingsnapshot.Entrypoint {
		return routingsnapshot.Entrypoint{
			ID: id, Revision: 1, Name: id, Aliases: []string{alias},
			Rules: []routingsnapshot.EntrypointRule{{
				ID: id + "-rule", Name: id, RecipeID: fixtureRecipeID, RecipeRevision: 1,
				Assignments: map[string]routingsnapshot.AssignmentSet{
					fixtureDecisionID: {Models: []routingsnapshot.Assignment{{
						ModelID: fixtureModelID, ModelRevision: 1, Weight: "1",
					}}},
				},
			}},
		}
	}
	return routingsnapshot.Bundle{
		NamespaceID: fixtureNamespaceID, Revision: 1, Currency: "USD",
		Models: []routingsnapshot.Model{{
			ID: fixtureModelID, Revision: 1,
			CatalogRevision: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
			Name:            "capacity/model", Capabilities: []string{"text"},
			Execution: routingsnapshot.ModelExecution{
				MaxRetries: 1, RequestTimeout: "30s", StreamTimeout: "60s",
			},
			Pricing: routingsnapshot.ModelPricing{
				InputCostPerMillionTokens: &inputPrice, OutputCostPerMillionTokens: &outputPrice,
			},
			Backends: []routingsnapshot.Backend{{
				ID: "capacity-backend", ProviderID: "openai-compatible",
				WireFormat: "openai.chat.v1", Origin: "https://capacity.example.invalid/v1",
				ProviderModelID: "capacity", Connection: routingsnapshot.BackendConnection{
					Path: "/chat/completions",
				}, Weight: "1",
			}},
		}},
		Recipes: []routingsnapshot.Recipe{{
			ID: fixtureRecipeID, Revision: 1, Name: "Capacity",
			Decisions: []routingsnapshot.Decision{{
				ID: fixtureDecisionID, Name: "Capacity",
				DispatchCardinality: routingsnapshot.DispatchCardinalitySingle,
			}},
			Document: json.RawMessage(`{"signals":[],"decisions":[]}`),
		}},
		Entrypoints: []routingsnapshot.Entrypoint{
			entrypoint(fixtureEntrypointA, "capacity/a"),
			entrypoint(fixtureEntrypointB, "capacity/b"),
		},
	}
}

func suffix(index int) string { return fmt.Sprintf("%06d", index) }
