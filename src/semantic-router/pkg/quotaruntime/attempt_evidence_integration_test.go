package quotaruntime

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quota"
)

func TestRedisAttemptEvidenceIsCrossReplicaContiguousAndIdempotent(t *testing.T) {
	client, partition := integrationRedis(t)
	first, second := attemptEvidenceEngines(t, client)
	preconditions, _, _ := seedAccessProjection(t, client, partition)
	rules := []RuleBinding{requestRule(t, "binding-attempts", "rpm", "12", time.Minute, 0)}
	admissionDigest := strings.Repeat("b", 64)
	admission, testRedisAttemptEvidenceIsCrossReplicaContiguousAndIdempotentErr := first.Admit(context.Background(), AdmissionRequest{
		Partition: partition, AdmissionID: "admission-attempts", Digest: admissionDigest,
		LeaseDuration: 2 * time.Minute, Preconditions: preconditions, Rules: rules,
	})
	if testRedisAttemptEvidenceIsCrossReplicaContiguousAndIdempotentErr != nil || !admission.Allowed() {
		t.Fatalf("Admit() = %+v, %v", admission, testRedisAttemptEvidenceIsCrossReplicaContiguousAndIdempotentErr)
	}
	planDigest := strings.Repeat("a", 64)
	journal := DispatchJournalRequest{
		Partition: partition, AdmissionID: "admission-attempts", AdmissionDigest: admissionDigest,
		DispatchID: "dispatch-1", Ordinal: 2, Digest: planDigest,
	}
	if _, err := first.JournalDispatch(context.Background(), journal); err != nil {
		t.Fatalf("JournalDispatch() error = %v", err)
	}
	reference := DispatchReference{
		Partition: partition, AdmissionID: journal.AdmissionID, AdmissionDigest: admissionDigest,
		DispatchID: journal.DispatchID, DispatchPlanDigest: planDigest,
		ModelID: "model-1", ModelRevision: 7, RequestDigest: testBackendRequestDigest("A"),
	}
	begin := BeginDispatchRequest{
		DispatchReference: reference, DispatchType: "primary", Ordinal: 2,
		Deadline: admission.ServerTime.Add(time.Minute), MaxAttempts: 3,
	}
	exerciseCrossReplicaAttemptSequence(t, first, second, begin)

	evidence, testRedisAttemptEvidenceIsCrossReplicaContiguousAndIdempotentErr := second.ReadAttemptEvidence(context.Background(), ReadAttemptEvidenceRequest{
		AttemptEvidenceReference: readAttemptReference(reference, 2),
	})
	if testRedisAttemptEvidenceIsCrossReplicaContiguousAndIdempotentErr != nil {
		t.Fatalf("ReadAttemptEvidence() error = %v", testRedisAttemptEvidenceIsCrossReplicaContiguousAndIdempotentErr)
	}
	if !evidence.Present || evidence.Revision != 5 ||
		evidence.Evidence.DispatchID != reference.DispatchID ||
		evidence.Evidence.Ordinal != 2 || evidence.Evidence.ModelID != "model-1" ||
		len(evidence.Evidence.Attempts) != 2 ||
		evidence.Evidence.Attempts[0].State != AttemptEvidenceKnownZero ||
		evidence.Evidence.Attempts[1].State != AttemptEvidenceResponseStarted ||
		evidence.Evidence.Attempts[1].StatusCode != http.StatusServiceUnavailable ||
		evidence.Evidence.Attempts[1].ErrorCode != "" ||
		!evidence.Evidence.Attempts[0].Finished || !evidence.Evidence.Attempts[1].Finished {
		t.Fatalf("attempt evidence = %+v", evidence.Evidence)
	}
	wrongReference := reference
	wrongReference.ModelRevision++
	if _, err := first.ReadAttemptEvidence(context.Background(), ReadAttemptEvidenceRequest{
		AttemptEvidenceReference: readAttemptReference(wrongReference, 2),
	}); !errors.Is(err, ErrConflict) {
		t.Fatalf("ReadAttemptEvidence() with changed model error = %v, want %v", err, ErrConflict)
	}

	assertAttemptEvidenceFinalization(t, client, first, second, partition, journal, admissionDigest, rules, evidence.Revision)
}

func exerciseCrossReplicaAttemptSequence(t *testing.T, first, second *RedisEngine, begin BeginDispatchRequest) {
	t.Helper()
	reference := begin.DispatchReference
	started, err := first.BeginDispatch(context.Background(), begin)
	if err != nil || started.Idempotent || started.StartedAt.IsZero() ||
		!started.Deadline.Equal(begin.Deadline) {
		t.Fatalf("BeginDispatch() = %+v, %v", started, err)
	}
	repeated, err := second.BeginDispatch(context.Background(), begin)
	if err != nil || !repeated.Idempotent || !repeated.StartedAt.Equal(started.StartedAt) {
		t.Fatalf("duplicate cross-replica BeginDispatch() = %+v, %v", repeated, err)
	}
	changed := begin
	changed.ModelRevision++
	if _, beginErr := second.BeginDispatch(context.Background(), changed); !errors.Is(beginErr, ErrConflict) {
		t.Fatalf("changed BeginDispatch() error = %v, want %v", beginErr, ErrConflict)
	}

	firstAttempt := BeginAttemptRequest{
		DispatchReference: reference, AttemptID: "dispatch-1:1", AttemptNumber: 1,
		BackendID: "backend-a", ProviderID: "provider-a",
	}
	wrongRequest := firstAttempt
	wrongRequest.RequestDigest = testBackendRequestDigest("B")
	if _, beginErr := first.BeginAttempt(context.Background(), wrongRequest); !errors.Is(beginErr, ErrConflict) {
		t.Fatalf("BeginAttempt() with changed request identity error = %v, want %v", beginErr, ErrConflict)
	}
	if _, beginErr := first.BeginAttempt(context.Background(), firstAttempt); beginErr != nil {
		t.Fatalf("BeginAttempt(1) error = %v", beginErr)
	}
	if _, beginErr := second.BeginAttempt(context.Background(), firstAttempt); !errors.Is(beginErr, ErrConflict) {
		t.Fatalf("duplicate BeginAttempt(1) error = %v, want %v", beginErr, ErrConflict)
	}
	secondAttempt := BeginAttemptRequest{
		DispatchReference: reference, AttemptID: "dispatch-1:2", AttemptNumber: 2,
		BackendID: "backend-b", ProviderID: "provider-b",
	}
	if _, beginErr := second.BeginAttempt(context.Background(), secondAttempt); !errors.Is(beginErr, ErrConflict) {
		t.Fatalf("BeginAttempt(2) before known-zero error = %v, want %v", beginErr, ErrConflict)
	}
	firstFinish := FinishAttemptRequest{
		DispatchReference: reference, AttemptID: firstAttempt.AttemptID, AttemptNumber: 1,
		BackendID: firstAttempt.BackendID, ProviderID: firstAttempt.ProviderID,
		State: AttemptEvidenceKnownZero, ErrorCode: "transport_error",
	}
	finished, err := second.FinishAttempt(context.Background(), firstFinish)
	if err != nil || finished.Idempotent || finished.CompletedAt.IsZero() {
		t.Fatalf("FinishAttempt(1) = %+v, %v", finished, err)
	}
	repeatedFinish, err := first.FinishAttempt(context.Background(), firstFinish)
	if err != nil || !repeatedFinish.Idempotent ||
		!repeatedFinish.CompletedAt.Equal(finished.CompletedAt) {
		t.Fatalf("duplicate FinishAttempt(1) = %+v, %v", repeatedFinish, err)
	}
	conflictingFinish := firstFinish
	conflictingFinish.ErrorCode = "different_error"
	if _, err := first.FinishAttempt(context.Background(), conflictingFinish); !errors.Is(err, ErrConflict) {
		t.Fatalf("conflicting FinishAttempt(1) error = %v, want %v", err, ErrConflict)
	}
	if _, err := second.BeginAttempt(context.Background(), secondAttempt); err != nil {
		t.Fatalf("BeginAttempt(2) error = %v", err)
	}
	secondFinish := FinishAttemptRequest{
		DispatchReference: reference, AttemptID: secondAttempt.AttemptID, AttemptNumber: 2,
		BackendID: secondAttempt.BackendID, ProviderID: secondAttempt.ProviderID,
		State: AttemptEvidenceResponseStarted, StatusCode: http.StatusServiceUnavailable,
	}
	if _, err := first.FinishAttempt(context.Background(), secondFinish); err != nil {
		t.Fatalf("FinishAttempt(2) error = %v", err)
	}
	thirdAttempt := BeginAttemptRequest{
		DispatchReference: reference, AttemptID: "dispatch-1:3", AttemptNumber: 3,
		BackendID: "backend-c", ProviderID: "provider-c",
	}
	if _, err := second.BeginAttempt(context.Background(), thirdAttempt); !errors.Is(err, ErrConflict) {
		t.Fatalf("retry after response-started error = %v, want %v", err, ErrConflict)
	}
}

func assertAttemptEvidenceFinalization(
	t *testing.T,
	client *redis.Client,
	first, second *RedisEngine,
	partition string,
	journal DispatchJournalRequest,
	admissionDigest string,
	rules []RuleBinding,
	evidenceRevision uint64,
) {
	t.Helper()
	finalization := FinalizationRequest{
		Partition: partition, AdmissionID: journal.AdmissionID, AdmissionDigest: admissionDigest,
		FinalizationDigest: "final-attempts", DispatchCount: 1,
		EvidenceRevision: evidenceRevision, Event: `{"admissionId":"admission-attempts"}`,
		EventEvidenceState: "known",
		Rules:              rules,
		Evidence:           map[quota.CounterIdentity]ActualEvidence{},
	}
	finalized, err := first.Finalize(context.Background(), finalization)
	if err != nil || finalized.Idempotent {
		t.Fatalf("Finalize() = %+v, %v", finalized, err)
	}
	keys, _ := newPartitionKeys(partition)
	if exists, existsErr := client.Exists(context.Background(), keys.attempts(journal.AdmissionID)).Result(); existsErr != nil || exists != 0 {
		t.Fatalf("attempt evidence exists after Finalize = %d, %v", exists, existsErr)
	}
	finalized, err = second.Finalize(context.Background(), finalization)
	if err != nil || !finalized.Idempotent {
		t.Fatalf("cross-replica duplicate Finalize() = %+v, %v", finalized, err)
	}
	meters, err := first.ReadMeters(context.Background(), MeterReadRequest{Partition: partition, Rules: rules})
	if err != nil || len(meters.Meters) != 1 || meters.Meters[0].Used != "1" {
		t.Fatalf("retry admission RPM meter = %+v, %v; want exactly one request", meters, err)
	}
}

func TestRedisAttemptEvidenceSurfacesCrashedAttemptAsUnknownForFencing(t *testing.T) {
	client, partition := integrationRedis(t)
	engine, _ := attemptEvidenceEngines(t, client)
	preconditions, _, _ := seedAccessProjection(t, client, partition)
	rules := []RuleBinding{tokenRule(t, "binding-crash", "tokens", "100", time.Minute, 0)}
	identity, _ := rules[0].Counter()
	admissionDigest := strings.Repeat("c", 64)
	admission, testRedisAttemptEvidenceSurfacesCrashedAttemptAsUnknownForFencingErr := engine.Admit(context.Background(), AdmissionRequest{
		Partition: partition, AdmissionID: "admission-crash", Digest: admissionDigest,
		LeaseDuration: time.Minute, Preconditions: preconditions, Rules: rules,
	})
	if testRedisAttemptEvidenceSurfacesCrashedAttemptAsUnknownForFencingErr != nil || !admission.Allowed() {
		t.Fatalf("Admit() = %+v, %v", admission, testRedisAttemptEvidenceSurfacesCrashedAttemptAsUnknownForFencingErr)
	}
	planDigest := strings.Repeat("d", 64)
	if _, err := engine.JournalDispatch(context.Background(), DispatchJournalRequest{
		Partition: partition, AdmissionID: "admission-crash", AdmissionDigest: admissionDigest,
		DispatchID: "dispatch-crash", Digest: planDigest,
	}); err != nil {
		t.Fatal(err)
	}
	reference := DispatchReference{
		Partition: partition, AdmissionID: "admission-crash", AdmissionDigest: admissionDigest,
		DispatchID: "dispatch-crash", DispatchPlanDigest: planDigest,
		ModelID: "model-crash", ModelRevision: 1, RequestDigest: testBackendRequestDigest("C"),
	}
	if _, err := engine.BeginDispatch(context.Background(), BeginDispatchRequest{
		DispatchReference: reference, DispatchType: "primary", Deadline: admission.ServerTime.Add(100 * time.Millisecond),
		MaxAttempts: 1,
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := engine.BeginAttempt(context.Background(), BeginAttemptRequest{
		DispatchReference: reference, AttemptID: "dispatch-crash:1", AttemptNumber: 1,
		BackendID: "backend-crash", ProviderID: "provider-crash",
	}); err != nil {
		t.Fatal(err)
	}
	time.Sleep(125 * time.Millisecond)
	evidence, testRedisAttemptEvidenceSurfacesCrashedAttemptAsUnknownForFencingErr := engine.ReadAttemptEvidence(context.Background(), ReadAttemptEvidenceRequest{
		AttemptEvidenceReference: readAttemptReference(reference, 0),
	})
	if testRedisAttemptEvidenceSurfacesCrashedAttemptAsUnknownForFencingErr != nil {
		t.Fatalf("ReadAttemptEvidence() after deadline error = %v", testRedisAttemptEvidenceSurfacesCrashedAttemptAsUnknownForFencingErr)
	}
	if len(evidence.Evidence.Attempts) != 1 || evidence.Evidence.Attempts[0].Finished ||
		evidence.Evidence.Attempts[0].State != AttemptEvidenceUnknown ||
		evidence.Evidence.Attempts[0].ErrorCode != unfinishedAttemptCode {
		t.Fatalf("crashed attempt evidence = %+v", evidence.Evidence.Attempts)
	}
	if _, err := engine.Finalize(context.Background(), FinalizationRequest{
		Partition: partition, AdmissionID: "admission-crash", AdmissionDigest: admissionDigest,
		FinalizationDigest: "final-crash", DispatchCount: 1,
		EvidenceRevision: evidence.Revision, Event: `{"admissionId":"admission-crash"}`,
		EventEvidenceState: "unknown",
		FenceID:            "fence-crash", Rules: rules,
		Evidence: map[quota.CounterIdentity]ActualEvidence{
			identity: {State: ActualEvidenceUnknown, Reason: unfinishedAttemptCode},
		},
	}); err != nil {
		t.Fatalf("Finalize(unknown crash) error = %v", err)
	}
	meters, testRedisAttemptEvidenceSurfacesCrashedAttemptAsUnknownForFencingErr := engine.ReadMeters(context.Background(), MeterReadRequest{Partition: partition, Rules: rules})
	if testRedisAttemptEvidenceSurfacesCrashedAttemptAsUnknownForFencingErr != nil || len(meters.Meters) != 1 ||
		meters.Meters[0].CapacityState != quota.CapacityFenced ||
		len(meters.Meters[0].ActiveFenceIDs) != 1 || meters.Meters[0].ActiveFenceIDs[0] != "fence-crash" {
		t.Fatalf("crash fence meters = %+v, %v", meters, testRedisAttemptEvidenceSurfacesCrashedAttemptAsUnknownForFencingErr)
	}
}

func readAttemptReference(reference DispatchReference, ordinal uint32) AttemptEvidenceReference {
	return AttemptEvidenceReference{
		Partition: reference.Partition, AdmissionID: reference.AdmissionID,
		AdmissionDigest: reference.AdmissionDigest, DispatchID: reference.DispatchID,
		Ordinal: ordinal, DispatchPlanDigest: reference.DispatchPlanDigest,
		ModelID: reference.ModelID, ModelRevision: reference.ModelRevision,
	}
}

func TestRedisFinalizationCompareAndSetsAttemptEvidenceRevision(t *testing.T) {
	client, partition := integrationRedis(t)
	engine, _ := attemptEvidenceEngines(t, client)
	preconditions, _, _ := seedAccessProjection(t, client, partition)
	rules := []RuleBinding{tokenRule(t, "binding-cas", "tokens", "100", time.Minute, 0)}
	identity, _ := rules[0].Counter()
	admissionDigest := strings.Repeat("7", 64)
	admission, testRedisFinalizationCompareAndSetsAttemptEvidenceRevisionErr := engine.Admit(context.Background(), AdmissionRequest{
		Partition: partition, AdmissionID: "admission-evidence-cas", Digest: admissionDigest,
		LeaseDuration: time.Minute, Preconditions: preconditions, Rules: rules,
	})
	if testRedisFinalizationCompareAndSetsAttemptEvidenceRevisionErr != nil || !admission.Allowed() {
		t.Fatalf("Admit() = %+v, %v", admission, testRedisFinalizationCompareAndSetsAttemptEvidenceRevisionErr)
	}
	planDigest := strings.Repeat("8", 64)
	if _, err := engine.JournalDispatch(context.Background(), DispatchJournalRequest{
		Partition: partition, AdmissionID: "admission-evidence-cas", AdmissionDigest: admissionDigest,
		DispatchID: "dispatch-evidence-cas", Digest: planDigest,
	}); err != nil {
		t.Fatal(err)
	}
	reference := DispatchReference{
		Partition: partition, AdmissionID: "admission-evidence-cas", AdmissionDigest: admissionDigest,
		DispatchID: "dispatch-evidence-cas", DispatchPlanDigest: planDigest,
		ModelID: "model-evidence-cas", ModelRevision: 1, RequestDigest: testBackendRequestDigest("CAS"),
	}
	missing, testRedisFinalizationCompareAndSetsAttemptEvidenceRevisionErr := engine.ReadAttemptEvidence(context.Background(), ReadAttemptEvidenceRequest{
		AttemptEvidenceReference: readAttemptReference(reference, 0),
	})
	if testRedisFinalizationCompareAndSetsAttemptEvidenceRevisionErr != nil || missing.Present || missing.Revision != 0 {
		t.Fatalf("missing evidence = %+v, %v", missing, testRedisFinalizationCompareAndSetsAttemptEvidenceRevisionErr)
	}
	if _, err := engine.BeginDispatch(context.Background(), BeginDispatchRequest{
		DispatchReference: reference, DispatchType: "primary", Deadline: admission.ServerTime.Add(30 * time.Second),
		MaxAttempts: 1,
	}); err != nil {
		t.Fatal(err)
	}
	started, testRedisFinalizationCompareAndSetsAttemptEvidenceRevisionErr := engine.ReadAttemptEvidence(context.Background(), ReadAttemptEvidenceRequest{
		AttemptEvidenceReference: readAttemptReference(reference, 0),
	})
	if testRedisFinalizationCompareAndSetsAttemptEvidenceRevisionErr != nil || !started.Present || started.Revision != 1 || len(started.Evidence.Attempts) != 0 {
		t.Fatalf("started evidence = %+v, %v", started, testRedisFinalizationCompareAndSetsAttemptEvidenceRevisionErr)
	}
	attempt := BeginAttemptRequest{
		DispatchReference: reference, AttemptID: "dispatch-evidence-cas:1", AttemptNumber: 1,
		BackendID: "backend-evidence-cas", ProviderID: "provider-evidence-cas",
	}
	if _, err := engine.BeginAttempt(context.Background(), attempt); err != nil {
		t.Fatal(err)
	}
	stale := FinalizationRequest{
		Partition: partition, AdmissionID: reference.AdmissionID, AdmissionDigest: admissionDigest,
		FinalizationDigest: "final-evidence-cas", DispatchCount: 1,
		EvidenceRevision: started.Revision, Event: `{"admissionId":"admission-evidence-cas"}`,
		EventEvidenceState: "unknown",
		FenceID:            "fence-evidence-cas", Rules: rules,
		Evidence: map[quota.CounterIdentity]ActualEvidence{
			identity: {State: ActualEvidenceUnknown, Reason: "attempt_unfinished"},
		},
	}
	if _, err := engine.Finalize(context.Background(), stale); !errors.Is(err, ErrEvidenceChanged) {
		t.Fatalf("stale Finalize() error = %v, want %v", err, ErrEvidenceChanged)
	}
	if _, err := engine.FinishAttempt(context.Background(), FinishAttemptRequest{
		DispatchReference: reference, AttemptID: attempt.AttemptID, AttemptNumber: 1,
		BackendID: attempt.BackendID, ProviderID: attempt.ProviderID,
		State: AttemptEvidenceUnknown, ErrorCode: "response_interrupted",
	}); err != nil {
		t.Fatal(err)
	}
	terminal, testRedisFinalizationCompareAndSetsAttemptEvidenceRevisionErr := engine.ReadAttemptEvidence(context.Background(), ReadAttemptEvidenceRequest{
		AttemptEvidenceReference: readAttemptReference(reference, 0),
	})
	if testRedisFinalizationCompareAndSetsAttemptEvidenceRevisionErr != nil || terminal.Revision != 3 || len(terminal.Evidence.Attempts) != 1 ||
		terminal.Evidence.Attempts[0].State != AttemptEvidenceUnknown {
		t.Fatalf("terminal evidence = %+v, %v", terminal, testRedisFinalizationCompareAndSetsAttemptEvidenceRevisionErr)
	}
	stale.EvidenceRevision = terminal.Revision
	stale.Evidence[identity] = ActualEvidence{State: ActualEvidenceUnknown, Reason: "response_interrupted"}
	if _, err := engine.Finalize(context.Background(), stale); err != nil {
		t.Fatalf("Finalize() with stable evidence error = %v", err)
	}
}

func TestRedisAttemptEvidenceBeginAttemptCASAcrossReplicas(t *testing.T) {
	client, partition := integrationRedis(t)
	first, second := attemptEvidenceEngines(t, client)
	preconditions, _, _ := seedAccessProjection(t, client, partition)
	admissionDigest := strings.Repeat("e", 64)
	admission, err := first.Admit(context.Background(), AdmissionRequest{
		Partition: partition, AdmissionID: "admission-race", Digest: admissionDigest,
		LeaseDuration: time.Minute, Preconditions: preconditions,
	})
	if err != nil || !admission.Allowed() {
		t.Fatalf("Admit() = %+v, %v", admission, err)
	}
	planDigest := strings.Repeat("f", 64)
	if _, err := first.JournalDispatch(context.Background(), DispatchJournalRequest{
		Partition: partition, AdmissionID: "admission-race", AdmissionDigest: admissionDigest,
		DispatchID: "dispatch-race", Digest: planDigest,
	}); err != nil {
		t.Fatal(err)
	}
	reference := DispatchReference{
		Partition: partition, AdmissionID: "admission-race", AdmissionDigest: admissionDigest,
		DispatchID: "dispatch-race", DispatchPlanDigest: planDigest,
		ModelID: "model-race", ModelRevision: 1, RequestDigest: testBackendRequestDigest("D"),
	}
	if _, err := first.BeginDispatch(context.Background(), BeginDispatchRequest{
		DispatchReference: reference, DispatchType: "primary", Deadline: admission.ServerTime.Add(30 * time.Second),
		MaxAttempts: 1,
	}); err != nil {
		t.Fatal(err)
	}

	const contenders = 32
	var allowed atomic.Int32
	var conflicts atomic.Int32
	unexpected := make(chan error, 1)
	var wait sync.WaitGroup
	wait.Add(contenders)
	for index := 0; index < contenders; index++ {
		go func(index int) {
			defer wait.Done()
			engine := first
			if index%2 == 1 {
				engine = second
			}
			_, err := engine.BeginAttempt(context.Background(), BeginAttemptRequest{
				DispatchReference: reference, AttemptID: "dispatch-race:1", AttemptNumber: 1,
				BackendID: "backend-race", ProviderID: "provider-race",
			})
			switch {
			case err == nil:
				allowed.Add(1)
			case errors.Is(err, ErrConflict):
				conflicts.Add(1)
			default:
				select {
				case unexpected <- fmt.Errorf("contender %d: %w", index, err):
				default:
				}
			}
		}(index)
	}
	wait.Wait()
	select {
	case err := <-unexpected:
		t.Fatal(err)
	default:
	}
	if allowed.Load() != 1 || conflicts.Load() != contenders-1 {
		t.Fatalf("concurrent BeginAttempt allowed/conflicts = %d/%d, want 1/%d",
			allowed.Load(), conflicts.Load(), contenders-1)
	}
}

func attemptEvidenceEngines(t *testing.T, client *redis.Client) (*RedisEngine, *RedisEngine) {
	t.Helper()
	first, err := NewRedisEngine(client, RedisEngineOptions{FinalizationMarkerTTL: time.Hour})
	if err != nil {
		t.Fatal(err)
	}
	second, err := NewRedisEngine(client, RedisEngineOptions{FinalizationMarkerTTL: time.Hour})
	if err != nil {
		t.Fatal(err)
	}
	return first, second
}

func testBackendRequestDigest(character string) string {
	digest := sha256.Sum256([]byte(character))
	return base64.RawURLEncoding.EncodeToString(digest[:])
}
