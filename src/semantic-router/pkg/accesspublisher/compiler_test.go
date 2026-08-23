package accesspublisher

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessprojection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/providercredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

func TestCompileCarriesLogicalKeyLifecycleIntoRuntimeProjection(t *testing.T) {
	state := validDesiredState(1, "100")
	expiresAt := fixtureTime.Add(30 * 24 * time.Hour)
	state.Keys[0].Key.ExpiresAt = &expiresAt
	publication, err := Compile(state)
	if err != nil {
		t.Fatal(err)
	}
	projection := publication.Access[0].Projection
	if projection.KeyStatus != accesscontrol.APIKeyStatusActive || projection.KeyExpiresAt == nil ||
		!projection.KeyExpiresAt.Equal(expiresAt) {
		t.Fatalf("compiled logical-key lifecycle = status %q, expires %v", projection.KeyStatus, projection.KeyExpiresAt)
	}
}

func TestCompileProducesDeterministicCoupledPublication(t *testing.T) {
	first := mustPublication(t, 1, "100")
	second := mustPublication(t, 1, "100")
	if first.ID != second.ID || first.Digest != second.Digest || first.Manifest.Digest != second.Manifest.Digest {
		t.Fatalf("publication is not deterministic: first=%+v second=%+v", first, second)
	}
	if len(first.Access) != 1 || len(first.Credentials) != 1 || first.Routing.Digest == "" {
		t.Fatalf("publication is incomplete: %+v", first)
	}
	if err := verifyPublication(first); err != nil {
		t.Fatalf("verifyPublication() error = %v", err)
	}
	if first.Access[0].Projection.Evaluate(
		accesscontrol.GrantResourceEntrypoint, "ep-chat", accesscontrol.GrantPermissionInvoke,
	) != accesscontrol.AccessDecisionAllow {
		t.Fatal("compiled publication lost its entrypoint grant")
	}
}

func TestCompilePreservesCanonicalEmptyRoutingResourceSet(t *testing.T) {
	state := validDesiredState(1, "100")
	state.Keys = nil
	state.Credentials = nil
	state.Routing.Models = nil
	state.Routing.Recipes = nil
	state.Routing.Entrypoints = nil

	publication, err := Compile(state)
	if err != nil {
		t.Fatal(err)
	}
	if publication.Routing.ResourceDigests == nil || publication.Manifest.RoutingResources == nil {
		t.Fatalf(
			"empty routing resource maps must remain canonical objects: routing=%#v manifest=%#v",
			publication.Routing.ResourceDigests,
			publication.Manifest.RoutingResources,
		)
	}
	if len(publication.Routing.ResourceDigests) != 0 || len(publication.Manifest.RoutingResources) != 0 {
		t.Fatalf(
			"empty routing publication contains resources: routing=%#v manifest=%#v",
			publication.Routing.ResourceDigests,
			publication.Manifest.RoutingResources,
		)
	}
	if err := verifyPublication(publication); err != nil {
		t.Fatalf("verify empty routing publication: %v", err)
	}
}

func TestCompileProjectsTenThousandIndependentAPIKeyPolicies(t *testing.T) {
	const keyCount = 10_000
	state := validDesiredState(11, "100")
	baseCandidate := state.Keys[0]
	baseCredential := state.Credentials[0].Credential
	state.Keys = make([]accessprojection.Candidate, 0, keyCount)
	state.Credentials = make([]CredentialCandidate, 0, keyCount)

	for index := range keyCount {
		suffix := fmt.Sprintf("%05d", index)
		keyID := accesscontrol.APIKeyID("key-scale-" + suffix)
		accessPolicyID := accesscontrol.AccessPolicyID("access-scale-" + suffix)
		ratePolicyID := accesscontrol.RateLimitPolicyID("rate-scale-" + suffix)
		accessBindingID := accesscontrol.PolicyBindingID("access-binding-scale-" + suffix)
		rateBindingID := accesscontrol.PolicyBindingID("rate-binding-scale-" + suffix)
		rateRuleID := accesscontrol.RateLimitRuleID("rate-rule-scale-" + suffix)

		candidate := baseCandidate
		candidate.Key.ID = keyID
		candidate.Key.Name = "Scale key " + suffix
		candidate.KeyAccessBindings = []accesscontrol.AccessPolicyBinding{{
			ID: accessBindingID, NamespaceID: state.Namespace.ID, Subject: candidate.Key.SubjectRef(),
			PolicyID: accessPolicyID, Status: accesscontrol.BindingStatusActive,
			Revision: accesscontrol.Revision(state.Revision),
		}}
		candidate.UserAccessBindings = nil
		candidate.TeamAccessBindings = nil
		candidate.AccessPolicies = map[accesscontrol.AccessPolicyID]accesscontrol.AccessPolicy{
			accessPolicyID: {
				NamespaceID: state.Namespace.ID, ID: accessPolicyID, DisplayName: "Scale access " + suffix,
				Status: accesscontrol.PolicyStatusActive, Revision: accesscontrol.Revision(state.Revision),
				CreatedAt: fixtureTime, UpdatedAt: fixtureTime,
				Grants: []accesscontrol.AccessPolicyGrant{{
					PolicyID: accessPolicyID,
					Resource: accesscontrol.GrantResource{
						Type: accesscontrol.GrantResourceEntrypoint, ID: "ep-chat",
					},
					Permission: accesscontrol.GrantPermissionInvoke, Effect: accesscontrol.GrantEffectAllow,
				}},
			},
		}
		candidate.KeyRateBindings = []accesscontrol.RateLimitBinding{{
			ID: rateBindingID, NamespaceID: state.Namespace.ID, Subject: candidate.Key.SubjectRef(),
			PolicyID: ratePolicyID, Mode: accesscontrol.RateBindingAllocation,
			QuotaPartitionID: state.Namespace.QuotaPartitionID, Status: accesscontrol.BindingStatusActive,
			Revision: accesscontrol.Revision(state.Revision),
		}}
		candidate.UserRateBindings = nil
		candidate.TeamRateBindings = nil
		limit := accesscontrol.QuotaValue(fmt.Sprintf("%d", index+1))
		candidate.RatePolicies = map[accesscontrol.RateLimitPolicyID]accesscontrol.RateLimitPolicy{
			ratePolicyID: {
				NamespaceID: state.Namespace.ID, ID: ratePolicyID, DisplayName: "Scale rate " + suffix,
				Status: accesscontrol.PolicyStatusActive, Revision: accesscontrol.Revision(state.Revision),
				CreatedAt: fixtureTime, UpdatedAt: fixtureTime,
				Rules: []accesscontrol.RateLimitRule{{
					ID: rateRuleID, PolicyID: ratePolicyID, Metric: accesscontrol.RateMetricRequests,
					Algorithm: accesscontrol.RateAlgorithmSlidingLog, Limit: limit, Window: time.Minute,
					Accounting: accesscontrol.RateAccountingRequest, Enforcement: accesscontrol.RateEnforcementEnforce,
				}},
			},
		}
		candidate.RoutingClaims = map[string]routingsnapshot.ClaimValue{
			"scale-key": {Kind: "string", String: suffix},
		}
		state.Keys = append(state.Keys, candidate)

		credential := baseCredential
		credential.ID = accesscontrol.CredentialVersionID("credential-scale-" + suffix)
		credential.APIKeyID = keyID
		credential.KID = "scale-kid-" + suffix
		credential.SecretHMAC = []byte(fmt.Sprintf("%032d", index))
		state.Credentials = append(state.Credentials, CredentialCandidate{
			Kind: CredentialKindAPIKey, Credential: credential,
		})
	}

	publication, err := Compile(state)
	if err != nil {
		t.Fatalf("Compile() 10,000-key publication: %v", err)
	}
	if len(publication.Access) != keyCount || len(publication.Credentials) != keyCount ||
		len(publication.Manifest.Access) != keyCount || len(publication.Manifest.Credentials) != keyCount {
		t.Fatalf("publication counts = access %d, credentials %d, manifest access %d, manifest credentials %d",
			len(publication.Access), len(publication.Credentials),
			len(publication.Manifest.Access), len(publication.Manifest.Credentials))
	}
	for _, index := range []int{0, keyCount / 2, keyCount - 1} {
		suffix := fmt.Sprintf("%05d", index)
		document := publication.Access[index]
		if document.KeyID != "key-scale-"+suffix || len(document.Projection.Grants) != 1 ||
			document.Projection.Grants[0].PolicyID != "access-scale-"+suffix ||
			len(document.Projection.RateBindings) != 1 ||
			document.Projection.RateBindings[0].PolicyID != "rate-scale-"+suffix ||
			len(document.Projection.RateBindings[0].Rules) != 1 ||
			document.Projection.RateBindings[0].Rules[0].Rule.WholeLimit == nil ||
			document.Projection.RateBindings[0].Rules[0].Rule.WholeLimit.String() != fmt.Sprintf("%d", index+1) {
			t.Fatalf("projection %d crossed key policy boundaries: key=%q projection=%+v",
				index, document.KeyID, document.Projection)
		}
		credential := publication.Credentials[index]
		if credential.PublicID != "scale-kid-"+suffix || credential.Projection.KeyID != document.KeyID {
			t.Fatalf("credential %d = kid %q key %q, want suffix %s",
				index, credential.PublicID, credential.Projection.KeyID, suffix)
		}
	}
	if err := verifyPublication(publication); err != nil {
		t.Fatalf("verifyPublication() 10,000-key publication: %v", err)
	}
}

func TestCompilePublishesOnlyEncryptedProviderCredentialMaterialDeterministically(t *testing.T) {
	state, _ := desiredStateWithProviderCredential(t, 1)
	first, err := Compile(state)
	if err != nil {
		t.Fatal(err)
	}
	second, err := Compile(state)
	if err != nil {
		t.Fatal(err)
	}
	if first.ID != second.ID || len(first.ProviderCredentials) != 1 ||
		first.Manifest.ProviderCredentials[providerFixtureCredentialID].Digest != first.ProviderCredentials[0].Digest {
		t.Fatalf("provider credential publication is not deterministic: first=%+v second=%+v", first, second)
	}
	payload, err := json.Marshal(first.ProviderCredentials[0])
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(payload), "provider-secret") {
		t.Fatal("provider credential plaintext leaked into publication")
	}
	if err := verifyPublication(first); err != nil {
		t.Fatalf("verify provider credential publication: %v", err)
	}
}

func TestVerifyPublicationRejectsProviderCredentialTampering(t *testing.T) {
	state, _ := desiredStateWithProviderCredential(t, 1)
	publication, err := Compile(state)
	if err != nil {
		t.Fatal(err)
	}
	publication.ProviderCredentials[0].Versions[0].Envelope.Ciphertext = append(
		[]byte(nil), publication.ProviderCredentials[0].Versions[0].Envelope.Ciphertext...,
	)
	publication.ProviderCredentials[0].Versions[0].Envelope.Ciphertext[0] ^= 0xff
	if err := verifyPublication(publication); !errors.Is(err, ErrStagedCorrupt) {
		t.Fatalf("tampered provider credential verification = %v", err)
	}
}

func TestVerifyProviderCredentialRejectsNonCanonicalVersionOrder(t *testing.T) {
	state, codec := desiredStateWithProviderCredential(t, 1)
	retiringID := "66666666-6666-4666-8666-666666666666"
	retiring, err := codec.Seal(
		state.ProviderCredentials[0].Credential,
		retiringID,
		[]byte("retiring-provider-secret"),
		fixtureTime,
	)
	if err != nil {
		t.Fatal(err)
	}
	retireAt := fixtureTime.Add(time.Hour)
	retiring.Status = providercredential.VersionRetiring
	retiring.ExpiresAt = &retireAt
	state.ProviderCredentials[0].Versions = append(state.ProviderCredentials[0].Versions, retiring)
	publication, err := Compile(state)
	if err != nil {
		t.Fatal(err)
	}
	document := publication.ProviderCredentials[0]
	document.Versions[0], document.Versions[1] = document.Versions[1], document.Versions[0]
	document.Digest = ""
	document.Digest, err = canonicalDigest(document)
	if err != nil {
		t.Fatal(err)
	}
	if err := verifyProviderCredentialDocument(document); !errors.Is(err, ErrStagedCorrupt) {
		t.Fatalf("non-canonical provider credential verification = %v", err)
	}
}

func TestCompileRejectsUnreferencedOrUnboundedProviderCredentials(t *testing.T) {
	t.Run("unreferenced", func(t *testing.T) {
		state, _ := desiredStateWithProviderCredential(t, 1)
		state.Routing.Models[0].Backends[0].ProviderCredentialID = ""
		if _, err := Compile(state); err == nil {
			t.Fatal("unreferenced provider credential unexpectedly compiled")
		}
	})
	t.Run("unbounded versions", func(t *testing.T) {
		state, _ := desiredStateWithProviderCredential(t, 1)
		version := state.ProviderCredentials[0].Versions[0]
		state.ProviderCredentials[0].Versions = make(
			[]providercredential.Version,
			maximumPublishedProviderCredentialVersions+1,
		)
		for index := range state.ProviderCredentials[0].Versions {
			state.ProviderCredentials[0].Versions[index] = version
		}
		if _, err := Compile(state); err == nil {
			t.Fatal("unbounded provider credential versions unexpectedly compiled")
		}
	})
}

func TestCompilePublishesInactiveProviderCredentialWithoutSecretMaterial(t *testing.T) {
	for _, status := range []providercredential.Status{
		providercredential.StatusDisabled,
		providercredential.StatusDeleted,
	} {
		t.Run(string(status), func(t *testing.T) {
			state, _ := desiredStateWithProviderCredential(t, 2)
			credential := &state.ProviderCredentials[0].Credential
			credential.Status = status
			credential.ActiveVersionID = nil
			credential.UpdatedAt = fixtureTime.Add(time.Minute)
			if status == providercredential.StatusDeleted {
				deletedAt := credential.UpdatedAt
				credential.DeletedAt = &deletedAt
			}
			state.ProviderCredentials[0].Versions = nil
			publication, err := Compile(state)
			if err != nil {
				t.Fatal(err)
			}
			document := publication.ProviderCredentials[0]
			if document.Credential.Status != status || len(document.Versions) != 0 {
				t.Fatalf("inactive provider credential document = %+v", document)
			}
		})
	}
}

func TestCompileRejectsCredentialVerifierOutsideRuntimeContract(t *testing.T) {
	state := validDesiredState(1, "100")
	state.Credentials[0].Credential.SecretHMAC = []byte("too-short")
	if _, err := Compile(state); err == nil {
		t.Fatal("short credential HMAC unexpectedly compiled")
	}
}

func TestCompileDelegatedCredentialCarriesExactAuthorityBinding(t *testing.T) {
	state := validDesiredState(1, "100")
	credential := state.Credentials[0].Credential
	credential.ID = "delegated-session-1"
	credential.KID = "delegated-public-1"
	credential.SecretHMAC = []byte("abcdefghijklmnopqrstuvwxyz012345")
	context := &accessprojection.DelegationContext{
		ManagementSessionID: "management-session-1", PrincipalID: "principal-1",
		DelegationEpoch: 9, UserID: "user-publisher", Audience: "vllm-sr-inference",
	}
	state.Credentials = append(state.Credentials, CredentialCandidate{
		Kind: CredentialKindDelegation, Credential: credential, Delegation: context,
	})
	publication, err := Compile(state)
	if err != nil {
		t.Fatal(err)
	}
	if len(publication.Credentials) != 2 {
		t.Fatalf("credential projections = %+v", publication.Credentials)
	}
	var delegated *CredentialDocument
	for index := range publication.Credentials {
		if publication.Credentials[index].Kind == CredentialKindDelegation {
			delegated = &publication.Credentials[index]
		}
	}
	if delegated == nil || delegated.Projection.ManagementSessionID != context.ManagementSessionID ||
		delegated.Projection.PrincipalID != context.PrincipalID ||
		delegated.Projection.DelegationEpoch != context.DelegationEpoch ||
		delegated.Projection.UserID != context.UserID || delegated.Projection.Audience != context.Audience {
		t.Fatalf("delegated projection = %+v", delegated)
	}
}

func TestCompileDisabledNamespacePublishesRestrictionBarrierAndNoKeys(t *testing.T) {
	state := validDesiredState(2, "100")
	state.Namespace.Status = accesscontrol.NamespaceStatusDisabled
	state.Keys = nil
	state.Credentials = nil
	publication, err := Compile(state)
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	if len(publication.Access) != 0 || len(publication.Credentials) != 0 || len(publication.BarrierHints) != 1 {
		t.Fatalf("disabled namespace publication = %+v", publication)
	}
	barrier := publication.BarrierHints[0]
	if barrier.Kind != "namespace" || barrier.ResourceID != string(state.Namespace.ID) {
		t.Fatalf("disabled namespace barrier = %+v", barrier)
	}
}

func TestDiffClassifiesQuotaExpansionAndRestriction(t *testing.T) {
	previous := mustPublication(t, 1, "100")
	expansion := mustPublication(t, 2, "200")
	barriers, err := Diff(previousDocuments(previous), expansion)
	if err != nil {
		t.Fatal(err)
	}
	if len(barriers) != 0 {
		t.Fatalf("quota expansion installed barriers: %+v", barriers)
	}
	restriction := mustPublication(t, 2, "50")
	barriers, err = Diff(previousDocuments(previous), restriction)
	if err != nil {
		t.Fatal(err)
	}
	if len(barriers) != 1 || barriers[0].Kind != "api_key" || barriers[0].ResourceID != "key-publisher" {
		t.Fatalf("quota restriction barriers = %+v", barriers)
	}
}

func previousDocuments(publication Publication) PreviousDocuments {
	result := PreviousDocuments{
		Manifest:            &publication.Manifest,
		Access:              make(map[string]AccessDocument, len(publication.Access)),
		Credentials:         make(map[string]CredentialDocument, len(publication.Credentials)),
		ProviderCredentials: make(map[string]ProviderCredentialDocument, len(publication.ProviderCredentials)),
		Routing:             &publication.Routing,
	}
	for _, document := range publication.Access {
		result.Access[document.KeyID] = document
	}
	for _, document := range publication.Credentials {
		result.Credentials[credentialIdentity(document.Kind, document.PublicID)] = document
	}
	for _, document := range publication.ProviderCredentials {
		result.ProviderCredentials[document.Credential.ID] = document
	}
	return result
}
