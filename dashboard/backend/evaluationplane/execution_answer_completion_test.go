package evaluationplane

import (
	"context"
	"encoding/json"
	"strings"
	"sync/atomic"
	"testing"
)

func TestBrokerFinalAnswerEligibilityPreservesObservedContent(t *testing.T) {
	for _, test := range []struct {
		name     string
		choice   string
		complete bool
	}{
		{"reasoning exhausted budget", `{"message":{"role":"assistant","content":null,"reasoning":"exact answer"},"finish_reason":"length"}`, false},
		{"reasoning without final answer", `{"message":{"role":"assistant","content":null,"reasoning_content":"exact answer"},"finish_reason":"stop"}`, false},
		{"partial final answer", `{"message":{"role":"assistant","content":"exact answer"},"finish_reason":"length"}`, false},
		{"complete final answer", `{"message":{"role":"assistant","content":"exact answer"},"finish_reason":"stop"}`, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			var calls atomic.Int64
			body := `{"choices":[` + test.choice + `],"usage":{"prompt_tokens":3,"completion_tokens":2}}`
			broker := newTypedChatBroker(t, &calls, body)
			models := broker.execute(context.Background(), workerBrokerRequest{
				ID: 1, Operation: workerBrokerListModels, Payload: json.RawMessage("null"), TimeoutMS: 1_000,
			})
			if !models.Success {
				t.Fatalf("model discovery failed: %+v", models)
			}
			request := workerBrokerRequest{
				ID: 2, Operation: workerBrokerRoutedChatCompletion, TrackID: "joint", CaseID: "case-1", AttemptID: "attempt-1",
				Payload: json.RawMessage(`{"model":"virtual-entrypoint","messages":[{"role":"user","content":"hello"}],"temperature":0,"stream":false}`), TimeoutMS: 1_000,
			}
			response := broker.execute(context.Background(), request)
			entry := broker.entries[2]
			if !response.Success || calls.Load() != 1 || entry.FinalAnswerComplete == nil || *entry.FinalAnswerComplete != test.complete {
				t.Fatalf("chat response or final answer observation = %+v, entry=%+v", response, entry)
			}
			if entry.ResponseContentDigest == nil || *entry.ResponseContentDigest != digestString("exact answer") {
				t.Fatalf("observed provider output digest changed: %+v", entry.ResponseContentDigest)
			}
			if err := validateStoredExecutionAttestationEntry(entry, 2); err != nil {
				t.Fatalf("stored completion evidence rejected: %v", err)
			}
			expected := "exact answer"
			grading := gradingCaseEvidence{ExpectedAnswer: &expected}
			quality := serverObservedAnswerQuality(entry, grading)
			if (quality != nil) != test.complete || (quality != nil && *quality != 1) {
				t.Fatalf("final answer quality = %v, complete=%v", quality, test.complete)
			}
			latency := float64(entry.LatencyMicroseconds) / 1000
			record := executionRecordEvidence{
				TrackID: "joint", CaseID: "case-1", AttemptID: "attempt-1", Status: "succeeded",
				SelectedArmID: entry.ArmID, SelectionStatus: entry.SelectionStatus, SelectionMethod: entry.SelectionMethod,
				Recipe: entry.Recipe, DecisionName: entry.DecisionName, Algorithm: entry.Algorithm,
				Success: &entry.Success, Quality: quality, LatencyMS: &latency,
				InputTokens: entry.InputTokens, OutputTokens: entry.OutputTokens,
				RuntimeCost: serverRuntimeCost(entry, broker.manifest.Target.Mixture.ModelArms),
			}
			messageDigest, err := canonicalMessageListDigest([]brokerMessage{{Role: "user", Content: json.RawMessage(`"hello"`)}})
			if err != nil {
				t.Fatal(err)
			}
			cases := visibleCaseSet{MessageDigests: map[string]string{"case-1": messageDigest}}
			if err := validateBrokerRecord(entry, record, cases, grading, broker.manifest.Target.Mixture.ModelArms, nil, broker.manifest.Seed); err != nil {
				t.Fatalf("matching worker grading rejected: %v", err)
			}
			if !test.complete {
				forged := 1.0
				record.Quality = &forged
				if err := validateBrokerRecord(entry, record, cases, grading, broker.manifest.Target.Mixture.ModelArms, nil, broker.manifest.Seed); err == nil || !strings.Contains(err.Error(), "quality differs") {
					t.Fatalf("incomplete answer accepted a perfect worker score: %v", err)
				}
				complete := true
				entry.FinalAnswerComplete = &complete
				changedReceipt, err := brokerEntryReceipt(entry)
				if err != nil || changedReceipt == entry.BrokerReceipt {
					t.Fatal("changed completion eligibility retained a valid broker receipt")
				}
			}
		})
	}
}

func TestStoredAttestationBindsFinalAnswerEligibility(t *testing.T) {
	manifest, attestation, _ := confidenceExecutionAttestationFixture(t)
	complete := false
	attestation.Entries[1].FinalAnswerComplete = &complete
	refreshExecutionAttestationDigests(t, &attestation)
	if err := validateExecutionAttestationIdentity(manifest.RunID, attestation); err != nil {
		t.Fatalf("new completion observation rejected: %v", err)
	}
	complete = true
	digest, err := executionAttestationDigest(attestation)
	if err != nil {
		t.Fatal(err)
	}
	attestation.Digest = digest
	if err := validateExecutionAttestationIdentity(manifest.RunID, attestation); err == nil || !strings.Contains(err.Error(), "broker receipt digest") {
		t.Fatalf("changed completion observation passed stored receipt validation: %v", err)
	}
}

func TestFinalAnswerEligibilityRemainsDistinctFromLegacyObservedOutput(t *testing.T) {
	answer := "exact answer"
	digest := digestString(answer)
	entry := executionAttestationEntry{Success: true, ResponseContentDigest: &digest}
	legacyReceipt, err := brokerEntryReceipt(entry)
	if err != nil {
		t.Fatal(err)
	}
	// Captured from the pre-observation receipt implementation. Keeping the
	// optional field absent must preserve existing stored receipt bytes.
	const preCompletionReceipt = "sha256:03c715cee80893a24725afdbb93415f0d4a477909ff8f866b24709c3c911d483"
	if legacyReceipt != preCompletionReceipt {
		t.Fatalf("legacy receipt = %s, want %s", legacyReceipt, preCompletionReceipt)
	}
	quality := serverObservedAnswerQuality(entry, gradingCaseEvidence{ExpectedAnswer: &answer})
	if quality == nil || *quality != 1 {
		t.Fatal("legacy observed-output grading changed")
	}
	incomplete := false
	entry.FinalAnswerComplete = &incomplete
	newReceipt, err := brokerEntryReceipt(entry)
	if err != nil || newReceipt == legacyReceipt {
		t.Fatal("new completion observation was not bound in the receipt")
	}
	if serverObservedAnswerQuality(entry, gradingCaseEvidence{ExpectedAnswer: &answer}) != nil {
		t.Fatal("new incomplete observation received legacy grading")
	}
	entry.FinalAnswerComplete = nil
	restoredReceipt, err := brokerEntryReceipt(entry)
	if err != nil || restoredReceipt != legacyReceipt {
		t.Fatal("legacy receipt changed after removing the optional observation")
	}
}
