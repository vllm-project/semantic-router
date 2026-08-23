package backendinvoker

import (
	"encoding/base64"
	"encoding/json"
	"strings"
	"testing"
	"time"
)

func TestDispatchCapabilityIsAudienceBoundAndTamperEvident(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	keyring := SigningKeyring{ActiveVersion: "v1", Keys: map[string][]byte{"v1": []byte(strings.Repeat("k", 32))}, MaxLifetime: time.Minute}
	capability := completeTestCapability(DispatchCapability{
		NamespaceID: "ns", QuotaPartition: "partition", RoutingRevision: 3,
		AdmissionID: "adm", AdmissionDigest: strings.Repeat("b", 64),
		Candidates:    []DispatchCandidate{testDispatchCandidate("dsp", "mdl", 2)},
		RequestDigest: RequestDigest("POST", "/v1/chat/completions", "", []byte(`{}`)), Method: "POST", Path: "/v1/chat/completions",
		Audience: "backend-invoker", IssuedAt: now.Unix(), ExpiresAt: now.Add(30 * time.Second).Unix(),
	})
	token, err := keyring.Sign(capability, now)
	if err != nil {
		t.Fatal(err)
	}
	verified, err := keyring.Verify(token, "backend-invoker", now.Add(time.Second))
	if err != nil || len(verified.Candidates) != 1 || verified.Candidates[0].DispatchID != capability.Candidates[0].DispatchID {
		t.Fatalf("verify = %#v, %v", verified, err)
	}
	if _, err := keyring.Verify(token, "another-audience", now); err == nil {
		t.Fatal("audience mismatch unexpectedly verified")
	}
	tampered := []byte(token)
	tampered[len(tampered)-1] ^= 1
	if _, err := keyring.Verify(string(tampered), "backend-invoker", now); err == nil {
		t.Fatal("tampered token unexpectedly verified")
	}
}

func TestDispatchCapabilityRejectsMalformedCandidateChains(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	keyring := SigningKeyring{ActiveVersion: "v1", Keys: map[string][]byte{"v1": []byte(strings.Repeat("k", 32))}, MaxLifetime: time.Minute}
	base := completeTestCapability(DispatchCapability{
		NamespaceID: "ns", QuotaPartition: "partition", RoutingRevision: 1,
		AdmissionID: "adm", AdmissionDigest: strings.Repeat("b", 64),
		Candidates: []DispatchCandidate{
			{DispatchID: "dispatch-0", DispatchType: "primary", Ordinal: 4, DispatchPlanDigest: strings.Repeat("a", 64), ModelID: "model-0", ModelRevision: 1, Priority: 0},
			{DispatchID: "dispatch-1", DispatchType: "primary", Ordinal: 5, DispatchPlanDigest: strings.Repeat("c", 64), ModelID: "model-1", ModelRevision: 1, Priority: 1},
		},
		Fallback:      FallbackPolicy{On: []FallbackTrigger{FallbackUnavailable, FallbackTimeout}},
		RequestDigest: RequestDigest("POST", "/v1/chat/completions", "", []byte(`{}`)),
		Method:        "POST", Path: "/v1/chat/completions", Audience: "backend-invoker",
		IssuedAt: now.Unix(), ExpiresAt: now.Add(30 * time.Second).Unix(),
	})
	for name, mutate := range map[string]func(*DispatchCapability){
		"duplicate dispatch": func(capability *DispatchCapability) {
			capability.Candidates[1].DispatchID = capability.Candidates[0].DispatchID
		},
		"duplicate model": func(capability *DispatchCapability) {
			capability.Candidates[1].ModelID = capability.Candidates[0].ModelID
		},
		"gapped ordinal": func(capability *DispatchCapability) { capability.Candidates[1].Ordinal++ },
		"reordered priority": func(capability *DispatchCapability) {
			capability.Candidates[0].Priority, capability.Candidates[1].Priority = 1, 0
		},
		"gapped priority": func(capability *DispatchCapability) { capability.Candidates[1].Priority = 2 },
		"duplicate trigger": func(capability *DispatchCapability) {
			capability.Fallback.On = []FallbackTrigger{FallbackUnavailable, FallbackUnavailable}
		},
		"unknown trigger": func(capability *DispatchCapability) { capability.Fallback.On = []FallbackTrigger{"other"} },
	} {
		t.Run(name, func(t *testing.T) {
			candidate := base
			candidate.Candidates = cloneDispatchCandidates(base.Candidates)
			candidate.Fallback = cloneFallbackPolicy(base.Fallback)
			mutate(&candidate)
			if _, err := keyring.Sign(candidate, now); err == nil {
				t.Fatal("malformed candidate chain unexpectedly signed")
			}
		})
	}
	oversized := base
	oversized.Candidates = make([]DispatchCandidate, maximumDispatchCandidates+1)
	if _, err := keyring.Sign(oversized, now); err == nil {
		t.Fatal("oversized candidate chain unexpectedly signed")
	}
}

func TestDispatchCapabilityReorderedSignedPayloadDoesNotVerify(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	keyring := SigningKeyring{ActiveVersion: "v1", Keys: map[string][]byte{"v1": []byte(strings.Repeat("k", 32))}, MaxLifetime: time.Minute}
	capability := completeTestCapability(DispatchCapability{
		NamespaceID: "ns", QuotaPartition: "partition", RoutingRevision: 1,
		AdmissionID: "adm", AdmissionDigest: strings.Repeat("b", 64),
		Candidates: []DispatchCandidate{
			{DispatchID: "dispatch-0", DispatchType: "primary", Ordinal: 0, DispatchPlanDigest: strings.Repeat("a", 64), ModelID: "model-0", ModelRevision: 1},
			{DispatchID: "dispatch-1", DispatchType: "primary", Ordinal: 1, DispatchPlanDigest: strings.Repeat("c", 64), ModelID: "model-1", ModelRevision: 1},
		},
		Fallback:      FallbackPolicy{On: []FallbackTrigger{FallbackUnavailable}},
		RequestDigest: RequestDigest("POST", "/v1/chat/completions", "", []byte(`{}`)), Method: "POST", Path: "/v1/chat/completions",
		Audience: "backend-invoker", IssuedAt: now.Unix(), ExpiresAt: now.Add(30 * time.Second).Unix(),
	})
	token, testDispatchCapabilityReorderedSignedPayloadDoesNotVerifyErr := keyring.Sign(capability, now)
	if testDispatchCapabilityReorderedSignedPayloadDoesNotVerifyErr != nil {
		t.Fatal(testDispatchCapabilityReorderedSignedPayloadDoesNotVerifyErr)
	}
	parts := strings.Split(token, ".")
	payload, testDispatchCapabilityReorderedSignedPayloadDoesNotVerifyErr := base64.RawURLEncoding.DecodeString(parts[2])
	if testDispatchCapabilityReorderedSignedPayloadDoesNotVerifyErr != nil {
		t.Fatal(testDispatchCapabilityReorderedSignedPayloadDoesNotVerifyErr)
	}
	if err := json.Unmarshal(payload, &capability); err != nil {
		t.Fatal(err)
	}
	capability.Candidates[0], capability.Candidates[1] = capability.Candidates[1], capability.Candidates[0]
	payload, testDispatchCapabilityReorderedSignedPayloadDoesNotVerifyErr = json.Marshal(capability)
	if testDispatchCapabilityReorderedSignedPayloadDoesNotVerifyErr != nil {
		t.Fatal(testDispatchCapabilityReorderedSignedPayloadDoesNotVerifyErr)
	}
	parts[2] = base64.RawURLEncoding.EncodeToString(payload)
	if _, err := keyring.Verify(strings.Join(parts, "."), capability.Audience, now); err == nil {
		t.Fatal("reordered signed candidate chain unexpectedly verified")
	}
}

func TestDispatchCapabilityRejectsExpiredAndOversizedLifetime(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	keyring := SigningKeyring{ActiveVersion: "v1", Keys: map[string][]byte{"v1": []byte(strings.Repeat("k", 32))}, MaxLifetime: time.Minute}
	base := completeTestCapability(DispatchCapability{NamespaceID: "ns", QuotaPartition: "partition", RoutingRevision: 1, AdmissionID: "adm", AdmissionDigest: strings.Repeat("b", 64), Candidates: []DispatchCandidate{testDispatchCandidate("dsp", "mdl", 1)}, RequestDigest: RequestDigest("POST", "/v1/chat/completions", "", []byte(`{}`)), Method: "POST", Path: "/v1/chat/completions", Audience: "backend-invoker", IssuedAt: now.Unix()})
	base.ExpiresAt = now.Add(2 * time.Minute).Unix()
	if _, err := keyring.Sign(base, now); err == nil {
		t.Fatal("oversized lifetime unexpectedly signed")
	}
	base.ExpiresAt = now.Add(time.Second).Unix()
	token, err := keyring.Sign(base, now)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := keyring.Verify(token, "backend-invoker", now.Add(2*time.Second)); err == nil {
		t.Fatal("expired capability unexpectedly verified")
	}
}
