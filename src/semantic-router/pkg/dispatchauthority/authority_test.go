package dispatchauthority

import (
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingcontext"
)

type recordingPreparer struct {
	prepared preparedIdentity
	err      error
	calls    int
	facts    accessruntime.DispatchFacts
}

type chainPreparer struct {
	base           preparedIdentity
	calls          int
	mutateCommonOn int
}

func (preparer *chainPreparer) prepare(
	_ accessruntime.Admission,
	facts accessruntime.DispatchFacts,
) (preparedIdentity, error) {
	preparer.calls++
	prepared := preparer.base
	prepared.dispatchID = facts.DispatchID
	prepared.ordinal = facts.Ordinal
	prepared.dispatchPlanDigest = facts.DispatchPlanDigest
	if preparer.calls == preparer.mutateCommonOn {
		prepared.routingRevision++
	}
	return prepared, nil
}

func (preparer *recordingPreparer) prepare(
	_ accessruntime.Admission,
	facts accessruntime.DispatchFacts,
) (preparedIdentity, error) {
	preparer.calls++
	preparer.facts = facts
	return preparer.prepared, preparer.err
}

func TestMeteredAuthorityIssuesPrimaryFromPreparedAdmissionOnly(t *testing.T) {
	now := time.Unix(1_900_000_000, 0).UTC()
	options := testIssuerOptions(now)
	verificationKeyring := cloneTestKeyring(options.Keyring)
	preparer := &recordingPreparer{prepared: testPreparedIdentity()}
	authority, err := newMeteredAuthority(preparer, options)
	if err != nil {
		t.Fatal(err)
	}
	defer authority.Close()

	// The authority-owned signer must not retain the caller's key slice.
	clear(options.Keyring.Keys["v1"])
	facts := accessruntime.DispatchFacts{
		DispatchID:         "caller-dispatch",
		Ordinal:            7,
		DispatchPlanDigest: strings.Repeat("c", 64),
	}
	body := []byte(`{"model":"logical-model"}`)
	token, err := authority.IssuePrimary(PrimaryIssueRequest{
		Dispatch:  facts,
		RequestID: "request-1",
		Final: FinalRequest{
			Model:      ModelIdentity{ID: "model-resource", Revision: 13},
			Method:     "POST",
			Path:       "/v1/chat/completions",
			Query:      "trace=off",
			WireFormat: "openai.chat.v1",
			Body:       body,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if preparer.calls != 1 || preparer.facts != facts {
		t.Fatalf("prepare calls = %d, facts = %+v", preparer.calls, preparer.facts)
	}
	capability, err := verificationKeyring.Verify(token, options.Audience, now)
	if err != nil {
		t.Fatal(err)
	}
	prepared := preparer.prepared
	if len(capability.Candidates) != 1 {
		t.Fatalf("capability candidates = %+v", capability.Candidates)
	}
	candidate := capability.Candidates[0]
	if capability.NamespaceID != prepared.namespaceID ||
		capability.QuotaPartition != prepared.quotaPartition ||
		capability.PublicationID != prepared.publicationID ||
		capability.RuntimeEpoch != prepared.runtimeEpoch ||
		capability.RoutingRevision != prepared.routingRevision ||
		capability.RoutingDigest != prepared.routingDigest ||
		capability.AdmissionID != prepared.admissionID ||
		capability.AdmissionDigest != prepared.admissionDigest ||
		candidate.DispatchID != prepared.dispatchID ||
		candidate.Ordinal != int(prepared.ordinal) ||
		candidate.DispatchPlanDigest != prepared.dispatchPlanDigest ||
		capability.RequestID != "request-1" || capability.WireFormat != "openai.chat.v1" ||
		candidate.DispatchType != primaryDispatchType {
		t.Fatalf("capability authority fields = %+v", capability)
	}
	if candidate.ModelID != "model-resource" || candidate.ModelRevision != 13 ||
		capability.RequestDigest != backendinvoker.RequestDigest("POST", "/v1/chat/completions", "trace=off", body) {
		t.Fatalf("capability request fields = %+v", capability)
	}
}

func TestMeteredAuthorityFailsClosedWhenAdmissionPreparationFails(t *testing.T) {
	now := time.Unix(1_900_000_000, 0).UTC()
	preparer := &recordingPreparer{
		prepared: testPreparedIdentity(),
		err:      errors.New("admission was modified"),
	}
	authority, err := newMeteredAuthority(preparer, testIssuerOptions(now))
	if err != nil {
		t.Fatal(err)
	}
	defer authority.Close()

	if _, err := authority.IssuePrimary(PrimaryIssueRequest{
		Dispatch:  accessruntime.DispatchFacts{DispatchID: "dispatch", DispatchPlanDigest: strings.Repeat("a", 64)},
		RequestID: "request-1",
		Final:     testFinalRequest(),
	}); err == nil || !strings.Contains(err.Error(), "prepare metered dispatch") {
		t.Fatalf("IssuePrimary() error = %v", err)
	}
	if _, err := authority.IssueGrant(GrantIssueRequest{
		Dispatch:  accessruntime.DispatchFacts{DispatchID: "dispatch", DispatchPlanDigest: strings.Repeat("a", 64)},
		RequestID: "request-1",
		Model:     testFinalRequest().Model,
	}); err == nil || !strings.Contains(err.Error(), "prepare metered dispatch grant") {
		t.Fatalf("IssueGrant() error = %v", err)
	}
	if preparer.calls != 2 {
		t.Fatalf("prepare calls = %d, want 2", preparer.calls)
	}
}

func TestMeteredAuthorityIssuesOneImmutableFallbackChain(t *testing.T) {
	now := time.Unix(1_900_000_000, 0).UTC()
	options := testIssuerOptions(now)
	verificationKeyring := cloneTestKeyring(options.Keyring)
	preparer := &chainPreparer{base: testPreparedIdentity()}
	authority, err := newMeteredAuthority(preparer, options)
	if err != nil {
		t.Fatal(err)
	}
	defer authority.Close()

	token, err := authority.IssueChain(MeteredChainIssueRequest{
		Candidates: []CandidateIssue{
			{Dispatch: accessruntime.DispatchFacts{DispatchID: "dispatch-0", Ordinal: 4, DispatchPlanDigest: strings.Repeat("a", 64)}, Model: ModelIdentity{ID: "model-0", Revision: 1}},
			{Dispatch: accessruntime.DispatchFacts{DispatchID: "dispatch-1", Ordinal: 5, DispatchPlanDigest: strings.Repeat("c", 64)}, Model: ModelIdentity{ID: "model-1", Revision: 2}, Priority: 1},
		},
		Fallback:  backendinvoker.FallbackPolicy{On: []backendinvoker.FallbackTrigger{backendinvoker.FallbackUnavailable, backendinvoker.FallbackTimeout}},
		RequestID: "request-1",
		Final:     ChainFinalRequest{Method: "POST", Path: "/v1/chat/completions", Query: "trace=off", WireFormat: "openai.chat.v1", Body: []byte(`{"model":"logical"}`)},
	})
	if err != nil {
		t.Fatal(err)
	}
	capability, err := verificationKeyring.Verify(token, options.Audience, now)
	if err != nil {
		t.Fatal(err)
	}
	if preparer.calls != 2 || len(capability.Candidates) != 2 ||
		capability.Candidates[0].DispatchID != "dispatch-0" ||
		capability.Candidates[1].ModelID != "model-1" ||
		capability.Candidates[1].Priority != 1 || capability.Query != "trace=off" {
		t.Fatalf("capability = %+v; prepare calls = %d", capability, preparer.calls)
	}
}

func TestMeteredAuthorityRejectsCandidateFromDifferentAdmissionGeneration(t *testing.T) {
	now := time.Unix(1_900_000_000, 0).UTC()
	preparer := &chainPreparer{base: testPreparedIdentity(), mutateCommonOn: 2}
	authority, err := newMeteredAuthority(preparer, testIssuerOptions(now))
	if err != nil {
		t.Fatal(err)
	}
	defer authority.Close()
	_, err = authority.IssueChain(MeteredChainIssueRequest{
		Candidates: []CandidateIssue{
			{Dispatch: accessruntime.DispatchFacts{DispatchID: "dispatch-0", DispatchPlanDigest: strings.Repeat("a", 64)}, Model: ModelIdentity{ID: "model-0", Revision: 1}},
			{Dispatch: accessruntime.DispatchFacts{DispatchID: "dispatch-1", Ordinal: 1, DispatchPlanDigest: strings.Repeat("c", 64)}, Model: ModelIdentity{ID: "model-1", Revision: 1}, Priority: 1},
		},
		Fallback:  backendinvoker.FallbackPolicy{On: []backendinvoker.FallbackTrigger{backendinvoker.FallbackUnavailable}},
		RequestID: "request-1", Final: ChainFinalRequest{Method: "POST", Path: "/v1/chat/completions", Body: []byte(`{}`)},
	})
	if err == nil || !strings.Contains(err.Error(), "one admission") {
		t.Fatalf("IssueChain() error = %v", err)
	}
}

func TestMeteredAuthorityGrantIsOpaqueExactAndOwnerBound(t *testing.T) {
	now := time.Unix(1_900_000_000, 0).UTC()
	options := testIssuerOptions(now)
	verificationKeyring := cloneTestKeyring(options.Keyring)
	authority, testMeteredAuthorityGrantIsOpaqueExactAndOwnerBoundErr := newMeteredAuthority(
		&recordingPreparer{prepared: testPreparedIdentity()},
		options,
	)
	if testMeteredAuthorityGrantIsOpaqueExactAndOwnerBoundErr != nil {
		t.Fatal(testMeteredAuthorityGrantIsOpaqueExactAndOwnerBoundErr)
	}
	defer authority.Close()

	grantToken, testMeteredAuthorityGrantIsOpaqueExactAndOwnerBoundErr := authority.IssueGrant(GrantIssueRequest{
		Dispatch: accessruntime.DispatchFacts{
			DispatchID:         "dispatch",
			Ordinal:            3,
			DispatchPlanDigest: strings.Repeat("a", 64),
		},
		RequestID: "request-1",
		Model:     ModelIdentity{ID: "model-resource", Revision: 13},
	})
	if testMeteredAuthorityGrantIsOpaqueExactAndOwnerBoundErr != nil {
		t.Fatal(testMeteredAuthorityGrantIsOpaqueExactAndOwnerBoundErr)
	}
	verified, testMeteredAuthorityGrantIsOpaqueExactAndOwnerBoundErr := authority.VerifyGrant(grantToken, testGrantVerification("request-1"))
	if testMeteredAuthorityGrantIsOpaqueExactAndOwnerBoundErr != nil {
		t.Fatal(testMeteredAuthorityGrantIsOpaqueExactAndOwnerBoundErr)
	}
	final := testFinalRequest()
	capabilityToken, testMeteredAuthorityGrantIsOpaqueExactAndOwnerBoundErr := authority.IssueFromGrant(verified, final)
	if testMeteredAuthorityGrantIsOpaqueExactAndOwnerBoundErr != nil {
		t.Fatal(testMeteredAuthorityGrantIsOpaqueExactAndOwnerBoundErr)
	}
	capability, testMeteredAuthorityGrantIsOpaqueExactAndOwnerBoundErr := verificationKeyring.Verify(capabilityToken, options.Audience, now)
	if testMeteredAuthorityGrantIsOpaqueExactAndOwnerBoundErr != nil {
		t.Fatal(testMeteredAuthorityGrantIsOpaqueExactAndOwnerBoundErr)
	}
	if len(capability.Candidates) != 1 || capability.Candidates[0].DispatchType != grantedDispatchType ||
		capability.Candidates[0].ModelID != final.Model.ID || capability.Candidates[0].ModelRevision != final.Model.Revision ||
		capability.RequestDigest != backendinvoker.RequestDigest(final.Method, final.Path, final.Query, final.Body) {
		t.Fatalf("nested capability = %+v", capability)
	}

	wrongModel := final
	wrongModel.Model.ID = "other-model"
	if _, err := authority.IssueFromGrant(verified, wrongModel); err == nil {
		t.Fatal("IssueFromGrant() accepted a different model")
	}
	if _, err := authority.IssueFromGrant(VerifiedGrant{}, final); err == nil {
		t.Fatal("IssueFromGrant() accepted a zero proof")
	}

	other, testMeteredAuthorityGrantIsOpaqueExactAndOwnerBoundErr := newMeteredAuthority(
		&recordingPreparer{prepared: testPreparedIdentity()},
		testIssuerOptions(now),
	)
	if testMeteredAuthorityGrantIsOpaqueExactAndOwnerBoundErr != nil {
		t.Fatal(testMeteredAuthorityGrantIsOpaqueExactAndOwnerBoundErr)
	}
	defer other.Close()
	if _, err := other.IssueFromGrant(verified, final); err == nil {
		t.Fatal("IssueFromGrant() accepted proof from another authority")
	}

	tampered := grantToken[:len(grantToken)-1] + "A"
	if _, err := authority.VerifyGrant(tampered, testGrantVerification("request-1")); err == nil {
		t.Fatal("VerifyGrant() accepted a tampered grant")
	}
}

func TestMeteredAuthorityConstructionAndCloseFailClosed(t *testing.T) {
	if _, err := NewMeteredAuthority(MeteredAuthorityOptions{}); err == nil {
		t.Fatal("NewMeteredAuthority() accepted a nil access runtime")
	}
	if _, err := newMeteredAuthority(nil, backendinvoker.CapabilityIssuerOptions{}); err == nil {
		t.Fatal("newMeteredAuthority() accepted a nil preparer")
	}
	if _, err := newMeteredAuthority(&recordingPreparer{}, backendinvoker.CapabilityIssuerOptions{}); err == nil {
		t.Fatal("newMeteredAuthority() accepted invalid issuer options")
	}

	now := time.Unix(1_900_000_000, 0).UTC()
	options := testIssuerOptions(now)
	sourceKey := options.Keyring.Keys["v1"]
	authority, err := newMeteredAuthority(
		&recordingPreparer{prepared: testPreparedIdentity()},
		options,
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := authority.Close(); err != nil {
		t.Fatal(err)
	}
	if err := authority.Close(); err != nil {
		t.Fatalf("second Close() error = %v", err)
	}
	if string(sourceKey) != strings.Repeat("k", 32) {
		t.Fatal("Close() modified caller-owned key material")
	}
	if _, err := authority.IssuePrimary(PrimaryIssueRequest{}); err == nil {
		t.Fatal("closed authority issued a primary capability")
	}
	if _, err := authority.IssueGrant(GrantIssueRequest{}); err == nil {
		t.Fatal("closed authority issued a grant")
	}
	if _, err := authority.VerifyGrant("grant", testGrantVerification("request-1")); err == nil {
		t.Fatal("closed authority verified a grant")
	}
	if _, err := authority.IssueFromGrant(VerifiedGrant{}, FinalRequest{}); err == nil {
		t.Fatal("closed authority issued from a grant")
	}

	var nilAuthority *MeteredAuthority
	if err := nilAuthority.Close(); err != nil {
		t.Fatalf("nil Close() error = %v", err)
	}
	if _, err := nilAuthority.IssuePrimary(PrimaryIssueRequest{}); err == nil {
		t.Fatal("nil authority issued a capability")
	}
}

func testPreparedIdentity() preparedIdentity {
	return preparedIdentity{
		namespaceID:        "namespace-1",
		quotaPartition:     "partition-1",
		publicationID:      "publication-1",
		runtimeEpoch:       2,
		routingRevision:    29,
		routingDigest:      strings.Repeat("d", 64),
		admissionID:        "admission-1",
		admissionDigest:    strings.Repeat("b", 64),
		dispatchID:         "prepared-dispatch",
		ordinal:            3,
		dispatchPlanDigest: strings.Repeat("a", 64),
	}
}

func testGrantVerification(requestID string) GrantVerificationRequest {
	return GrantVerificationRequest{
		Generation: routingcontext.Generation{
			NamespaceID:      "namespace-1",
			QuotaPartition:   "partition-1",
			PublicationID:    "publication-1",
			RuntimeEpoch:     2,
			SnapshotRevision: 29,
			RoutingDigest:    strings.Repeat("d", 64),
		},
		RequestID: requestID,
	}
}

func testFinalRequest() FinalRequest {
	return FinalRequest{
		Model:      ModelIdentity{ID: "model-resource", Revision: 13},
		Method:     "POST",
		Path:       "/v1/chat/completions",
		WireFormat: "openai.chat.v1",
		Body:       []byte(`{"model":"logical-model"}`),
	}
}

func testIssuerOptions(now time.Time) backendinvoker.CapabilityIssuerOptions {
	return backendinvoker.CapabilityIssuerOptions{
		Audience: "vllm-sr.backend-dispatch",
		Keyring: backendinvoker.SigningKeyring{
			ActiveVersion: "v1",
			Keys:          map[string][]byte{"v1": []byte(strings.Repeat("k", 32))},
			MaxLifetime:   time.Minute,
		},
		Lifetime: 30 * time.Second,
		Now:      func() time.Time { return now },
	}
}

func cloneTestKeyring(source backendinvoker.SigningKeyring) backendinvoker.SigningKeyring {
	result := backendinvoker.SigningKeyring{
		ActiveVersion: source.ActiveVersion,
		Keys:          make(map[string][]byte, len(source.Keys)),
		MaxLifetime:   source.MaxLifetime,
	}
	for version, key := range source.Keys {
		result.Keys[version] = append([]byte(nil), key...)
	}
	return result
}
