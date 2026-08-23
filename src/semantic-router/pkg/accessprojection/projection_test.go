package accessprojection

import (
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

func validCandidate() Candidate {
	now := time.Date(2026, 8, 22, 1, 2, 3, 0, time.UTC)
	namespace := accesscontrol.Namespace{
		ID: "ns-1", Name: "default", QuotaPartitionID: "partition-1", BillingCurrency: "USD",
		Status: accesscontrol.NamespaceStatusActive, Revision: 1, RuntimeEpoch: 1, CreatedAt: now, UpdatedAt: now,
	}
	user := accesscontrol.User{
		NamespaceID: namespace.ID, ID: "user-1", Email: "user@example.com", DisplayName: "User",
		Status: accesscontrol.UserStatusActive, CreatedAt: now, UpdatedAt: now,
	}
	team := accesscontrol.Team{
		NamespaceID: namespace.ID, ID: "team-1", Name: "Team", Status: accesscontrol.TeamStatusActive,
		CreatedAt: now, UpdatedAt: now,
	}
	membership := accesscontrol.TeamMembership{
		NamespaceID: namespace.ID, TeamID: team.ID, UserID: user.ID, Role: accesscontrol.TeamRoleMember,
		Status: accesscontrol.MembershipStatusActive, CreatedAt: now, UpdatedAt: now,
	}
	key := accesscontrol.APIKey{
		NamespaceID: namespace.ID, ID: "key-1", Name: "User", Owner: user.SubjectRef(), ContextTeamID: team.ID,
		Status: accesscontrol.APIKeyStatusActive, PolicyEpoch: 1, DelegationEpoch: 1, Revision: 1,
		CreatedAt: now, UpdatedAt: now,
	}
	accessPolicy := accesscontrol.AccessPolicy{
		NamespaceID: namespace.ID, ID: "access-user", DisplayName: "User access", Status: accesscontrol.PolicyStatusActive,
		Revision: 1, CreatedAt: now, UpdatedAt: now,
		Grants: []accesscontrol.AccessPolicyGrant{
			{PolicyID: "access-user", Resource: accesscontrol.GrantResource{Type: accesscontrol.GrantResourceEntrypoint, ID: "ep-chat"}, Permission: accesscontrol.GrantPermissionDiscover, Effect: accesscontrol.GrantEffectAllow},
			{PolicyID: "access-user", Resource: accesscontrol.GrantResource{Type: accesscontrol.GrantResourceEntrypoint, ID: "ep-chat"}, Permission: accesscontrol.GrantPermissionInvoke, Effect: accesscontrol.GrantEffectAllow},
		},
	}
	ratePolicy := accesscontrol.RateLimitPolicy{
		NamespaceID: namespace.ID, ID: "rate-user", DisplayName: "User budget", Status: accesscontrol.PolicyStatusActive,
		Revision: 1, CreatedAt: now, UpdatedAt: now,
		Rules: []accesscontrol.RateLimitRule{{
			ID: "rule-rpm", PolicyID: "rate-user", Metric: accesscontrol.RateMetricRequests,
			Algorithm: accesscontrol.RateAlgorithmSlidingLog, Limit: "12", Window: time.Minute,
			Accounting: accesscontrol.RateAccountingRequest, Enforcement: accesscontrol.RateEnforcementEnforce,
		}},
	}
	return Candidate{
		Revision: 5, Namespace: namespace, Key: key,
		Relationships: accesscontrol.APIKeyRelationships{OwnerUser: &user, ContextTeam: &team, ContextMembership: &membership},
		UserAccessBindings: []accesscontrol.AccessPolicyBinding{{
			ID: "access-binding-user", NamespaceID: namespace.ID, Subject: user.SubjectRef(), PolicyID: accessPolicy.ID,
			Status: accesscontrol.BindingStatusActive, Revision: 1,
		}},
		AccessPolicies: map[accesscontrol.AccessPolicyID]accesscontrol.AccessPolicy{accessPolicy.ID: accessPolicy},
		TeamRateBindings: []accesscontrol.RateLimitBinding{{
			ID: "rate-binding-team", NamespaceID: namespace.ID, Subject: team.SubjectRef(), PolicyID: ratePolicy.ID,
			Mode: accesscontrol.RateBindingAllocation, QuotaPartitionID: namespace.QuotaPartitionID,
			Status: accesscontrol.BindingStatusActive, Revision: 1,
		}},
		RatePolicies:  map[accesscontrol.RateLimitPolicyID]accesscontrol.RateLimitPolicy{ratePolicy.ID: ratePolicy},
		RoutingClaims: map[string]routingsnapshot.ClaimValue{"tier": {Kind: "string", String: "free"}},
	}
}

func TestCompileProjectionPreservesInheritanceAndCounterOwnership(t *testing.T) {
	projection, err := Compile(validCandidate(), CompileOptions{})
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	if projection.AccessSource != accesscontrol.InheritanceLayerUser || projection.UserID != "user-1" || projection.TeamID != "team-1" {
		t.Fatalf("identity/access inheritance = %+v", projection)
	}
	if len(projection.RateBindings) != 1 || projection.RateBindings[0].BindingID != "rate-binding-team" ||
		projection.RateBindings[0].Source != accesscontrol.InheritanceLayerTeam {
		t.Fatalf("quota inheritance = %+v", projection.RateBindings)
	}
	if projection.Evaluate(accesscontrol.GrantResourceEntrypoint, "ep-chat", accesscontrol.GrantPermissionInvoke) != accesscontrol.AccessDecisionAllow {
		t.Fatal("published entrypoint invoke grant did not allow")
	}
	if projection.Evaluate(accesscontrol.GrantResourceModel, "mdl-hidden", accesscontrol.GrantPermissionInvoke) != accesscontrol.AccessDecisionDeny {
		t.Fatal("unmatched direct model did not deny")
	}
}

func TestCompileProjectionIsDeterministic(t *testing.T) {
	first, err := Compile(validCandidate(), CompileOptions{})
	if err != nil {
		t.Fatal(err)
	}
	second, err := Compile(validCandidate(), CompileOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if first.Digest == "" || first.Digest != second.Digest {
		t.Fatalf("digest mismatch %q != %q", first.Digest, second.Digest)
	}
}

func TestCompileProjectionKeyOverrideAndDenyPrecedence(t *testing.T) {
	candidate := validCandidate()
	keyPolicy := candidate.AccessPolicies["access-user"]
	keyPolicy.ID = "access-key"
	keyPolicy.DisplayName = "Key override"
	for index := range keyPolicy.Grants {
		keyPolicy.Grants[index].PolicyID = keyPolicy.ID
	}
	keyPolicy.Grants = append(keyPolicy.Grants, accesscontrol.AccessPolicyGrant{
		PolicyID: keyPolicy.ID, Resource: accesscontrol.GrantResource{Type: accesscontrol.GrantResourceEntrypoint, ID: "ep-chat"},
		Permission: accesscontrol.GrantPermissionInvoke, Effect: accesscontrol.GrantEffectDeny,
	})
	candidate.AccessPolicies[keyPolicy.ID] = keyPolicy
	candidate.KeyAccessBindings = []accesscontrol.AccessPolicyBinding{{
		ID: "access-binding-key", NamespaceID: candidate.Namespace.ID, Subject: candidate.Key.SubjectRef(),
		PolicyID: keyPolicy.ID, Status: accesscontrol.BindingStatusActive, Revision: 1,
	}}
	projection, err := Compile(candidate, CompileOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if projection.AccessSource != accesscontrol.InheritanceLayerKey {
		t.Fatalf("access source = %s", projection.AccessSource)
	}
	if projection.Evaluate(accesscontrol.GrantResourceEntrypoint, "ep-chat", accesscontrol.GrantPermissionInvoke) != accesscontrol.AccessDecisionDeny {
		t.Fatal("deny did not take precedence")
	}
}

func TestCompileProjectionRejectsForeignLayerSubject(t *testing.T) {
	candidate := validCandidate()
	candidate.TeamRateBindings[0].Subject.ID = "other-team"
	if _, err := Compile(candidate, CompileOptions{}); err == nil {
		t.Fatal("foreign team binding unexpectedly compiled")
	}
}

func TestCompileCredentialStripsRevealEnvelope(t *testing.T) {
	now := time.Date(2026, 8, 22, 1, 2, 3, 0, time.UTC)
	credential := accesscontrol.CredentialVersion{
		ID: "cred-1", APIKeyID: "key-1", KID: "public-kid-1", SecretHMAC: []byte("hmac"), PepperVersion: "pepper-1",
		SecretCiphertext: []byte("ciphertext"), CiphertextNonce: []byte("nonce"), KEKVersion: "kek-1",
		Status: accesscontrol.CredentialStatusActive, NotBefore: now, CreatedAt: now,
	}
	projection, err := CompileCredential("api_key", credential, nil)
	if err != nil {
		t.Fatal(err)
	}
	if projection.KID != credential.KID || projection.PepperVersion != credential.PepperVersion {
		t.Fatalf("credential projection = %+v", projection)
	}
}

func TestCompileDelegatedCredentialRequiresCompleteAuthorityContext(t *testing.T) {
	now := time.Date(2026, 8, 22, 1, 2, 3, 0, time.UTC)
	expires := now.Add(time.Hour)
	credential := accesscontrol.CredentialVersion{
		ID: "delegation-1", APIKeyID: "key-1", KID: "delegation-public-1",
		SecretHMAC: []byte("01234567890123456789012345678901"), PepperVersion: "pepper-1",
		Status: accesscontrol.CredentialStatusActive, NotBefore: now, ExpiresAt: &expires, CreatedAt: now,
	}
	if _, err := CompileCredential("delegation", credential, nil); err == nil {
		t.Fatal("delegation without authority context compiled")
	}
	context := &DelegationContext{
		ManagementSessionID: "management-session-1", PrincipalID: "principal-1",
		DelegationEpoch: 4, UserID: "user-1", TeamID: "team-1", Audience: "vllm-sr-inference",
	}
	projection, err := CompileCredential("delegation", credential, context)
	if err != nil {
		t.Fatal(err)
	}
	if projection.Kind != "delegation" || projection.ManagementSessionID != context.ManagementSessionID ||
		projection.PrincipalID != context.PrincipalID || projection.DelegationEpoch != context.DelegationEpoch ||
		projection.UserID != context.UserID || projection.TeamID != context.TeamID ||
		projection.Audience != context.Audience {
		t.Fatalf("delegation projection = %+v", projection)
	}
	if _, err := CompileCredential("api_key", credential, context); err == nil {
		t.Fatal("direct API key credential accepted delegation context")
	}
}

func TestCompileProjectsLogicalKeyExpiryAndDeterministicCalendarSchedule(t *testing.T) {
	candidate := validCandidate()
	expiresAt := candidate.Key.CreatedAt.Add(24 * time.Hour)
	candidate.Key.ExpiresAt = &expiresAt
	policy := candidate.RatePolicies["rate-user"]
	policy.Rules[0] = accesscontrol.RateLimitRule{
		ID: "rule-daily", PolicyID: policy.ID, Metric: accesscontrol.RateMetricRequests,
		Algorithm: accesscontrol.RateAlgorithmCalendarWindow, Limit: "1000",
		CalendarPeriod: accesscontrol.CalendarPeriodDay, Timezone: "America/Los_Angeles",
		Accounting: accesscontrol.RateAccountingRequest, Enforcement: accesscontrol.RateEnforcementEnforce,
	}
	candidate.RatePolicies[policy.ID] = policy
	start := time.Date(2026, 8, 22, 1, 2, 3, 0, time.UTC)
	options := CompileOptions{CalendarScheduleStart: start}
	first, err := Compile(candidate, options)
	if err != nil {
		t.Fatal(err)
	}
	second, err := Compile(candidate, options)
	if err != nil {
		t.Fatal(err)
	}
	if first.Digest != second.Digest || first.KeyExpiresAt == nil || !first.KeyExpiresAt.Equal(expiresAt) {
		t.Fatalf("logical key/schedule projection is unstable: first=%+v second=%+v", first, second)
	}
	projected := first.RateBindings[0].Rules[0]
	if projected.TZDBVersion != EmbeddedTZDBVersion || len(projected.CalendarSchedule) < 500 {
		t.Fatalf("calendar projection = %+v", projected)
	}
	if projected.CalendarSchedule[0].Start.After(start) ||
		projected.CalendarSchedule[len(projected.CalendarSchedule)-1].End.Before(start.AddDate(0, 18, 0)) {
		t.Fatalf("calendar schedule does not cover 18 months")
	}
}
