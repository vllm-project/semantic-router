package backend

import (
	"testing"
	"time"
)

func TestSelectBackendCandidatePrefersFreshLowestQueue(t *testing.T) {
	base := time.Unix(300, 0)
	store := newStoreWithClock(5*time.Second, func() time.Time { return base })
	queueHigh := 8
	queueLow := 2
	activeLow := 1
	activeHigh := 7
	for _, telemetry := range []BackendTelemetry{
		{
			Identity:       BackendIdentity{BackendID: "backend-a", ModelName: "qwen3-8b", ReplicaID: "a-0"},
			QueueDepth:     &queueHigh,
			ActiveRequests: &activeLow,
			Health:         HealthStateHealthy,
			CollectedAt:    base,
		},
		{
			Identity:       BackendIdentity{BackendID: "backend-b", ModelName: "qwen3-8b", ReplicaID: "b-0"},
			QueueDepth:     &queueLow,
			ActiveRequests: &activeHigh,
			Health:         HealthStateHealthy,
			CollectedAt:    base,
		},
	} {
		if err := store.Upsert(telemetry); err != nil {
			t.Fatalf("Upsert returned error: %v", err)
		}
	}

	result := SelectBackendCandidate("qwen3-8b", []BackendCandidate{
		{BackendID: "backend-a", Weight: 100},
		{BackendID: "backend-b", Weight: 1},
	}, store)
	if result.FailOpen {
		t.Fatalf("expected backend selection, got fail-open result %#v", result)
	}
	if result.SelectedBackendID != "backend-b" || result.SelectedReplicaID != "b-0" {
		t.Fatalf("expected backend-b/b-0, got %#v", result)
	}
	if result.Reason != PolicyReasonSelectedFreshTelemetry {
		t.Fatalf("Reason = %q, want %q", result.Reason, PolicyReasonSelectedFreshTelemetry)
	}
	if result.Diagnostics.CandidateCount != 2 || result.Diagnostics.FreshCandidateCount != 2 {
		t.Fatalf("unexpected diagnostics counts: %#v", result.Diagnostics)
	}
}

func TestSelectBackendCandidateTieBreaksWithActiveRequestsWeightThenBackendID(t *testing.T) {
	base := time.Unix(400, 0)
	store := newStoreWithClock(5*time.Second, func() time.Time { return base })
	queue := 1
	activeBusy := 5
	activeIdle := 1

	samples := []BackendTelemetry{
		{
			Identity:       BackendIdentity{BackendID: "backend-a", ModelName: "qwen3-8b"},
			QueueDepth:     &queue,
			ActiveRequests: &activeBusy,
			Health:         HealthStateHealthy,
			CollectedAt:    base,
		},
		{
			Identity:       BackendIdentity{BackendID: "backend-b", ModelName: "qwen3-8b"},
			QueueDepth:     &queue,
			ActiveRequests: &activeIdle,
			Health:         HealthStateHealthy,
			CollectedAt:    base,
		},
		{
			Identity:       BackendIdentity{BackendID: "backend-c", ModelName: "qwen3-8b"},
			QueueDepth:     &queue,
			ActiveRequests: &activeIdle,
			Health:         HealthStateHealthy,
			CollectedAt:    base,
		},
	}
	for _, sample := range samples {
		if err := store.Upsert(sample); err != nil {
			t.Fatalf("Upsert returned error: %v", err)
		}
	}

	result := SelectBackendCandidate("qwen3-8b", []BackendCandidate{
		{BackendID: "backend-a", Weight: 100},
		{BackendID: "backend-b", Weight: 10},
		{BackendID: "backend-c", Weight: 20},
	}, store)
	if result.SelectedBackendID != "backend-c" {
		t.Fatalf("expected higher-weight idle backend-c, got %#v", result)
	}

	result = SelectBackendCandidate("qwen3-8b", []BackendCandidate{
		{BackendID: "backend-b", Weight: 20},
		{BackendID: "backend-c", Weight: 20},
	}, store)
	if result.SelectedBackendID != "backend-b" {
		t.Fatalf("expected stable backend ID tie-break to choose backend-b, got %#v", result)
	}
}

func TestSelectBackendCandidateFailOpenReasons(t *testing.T) {
	base := time.Unix(500, 0)

	t.Run("no candidates", func(t *testing.T) {
		result := SelectBackendCandidate("qwen3-8b", nil, NewStore(DefaultTelemetryTTL))
		assertFailOpenReason(t, result, FallbackReasonNoBackendCandidates)
	})

	t.Run("missing telemetry", func(t *testing.T) {
		result := SelectBackendCandidate("qwen3-8b", []BackendCandidate{{BackendID: "missing"}}, NewStore(DefaultTelemetryTTL))
		assertFailOpenReason(t, result, FallbackReasonMissingTelemetry)
	})

	t.Run("stale telemetry", func(t *testing.T) {
		store := newStoreWithClock(5*time.Second, func() time.Time { return base })
		if err := store.Upsert(BackendTelemetry{
			Identity:    BackendIdentity{BackendID: "stale", ModelName: "qwen3-8b"},
			Health:      HealthStateHealthy,
			CollectedAt: base.Add(-6 * time.Second),
		}); err != nil {
			t.Fatalf("Upsert returned error: %v", err)
		}
		result := SelectBackendCandidate("qwen3-8b", []BackendCandidate{{BackendID: "stale"}}, store)
		assertFailOpenReason(t, result, FallbackReasonStaleTelemetry)
		if result.Diagnostics.TelemetryFresh {
			t.Fatal("expected stale telemetry to report TelemetryFresh=false")
		}
		if result.Diagnostics.TelemetryAgeMS != (6 * time.Second).Milliseconds() {
			t.Fatalf("TelemetryAgeMS = %d, want 6000", result.Diagnostics.TelemetryAgeMS)
		}
	})

	t.Run("all unhealthy", func(t *testing.T) {
		store := newStoreWithClock(5*time.Second, func() time.Time { return base })
		if err := store.Upsert(BackendTelemetry{
			Identity:    BackendIdentity{BackendID: "unhealthy", ModelName: "qwen3-8b"},
			Health:      HealthStateUnhealthy,
			CollectedAt: base,
		}); err != nil {
			t.Fatalf("Upsert returned error: %v", err)
		}
		result := SelectBackendCandidate("qwen3-8b", []BackendCandidate{{BackendID: "unhealthy"}}, store)
		assertFailOpenReason(t, result, FallbackReasonAllUnhealthy)
		if !result.Diagnostics.TelemetryFresh {
			t.Fatal("expected fresh unhealthy telemetry to report TelemetryFresh=true")
		}
		if result.Diagnostics.UnhealthyCount != 1 {
			t.Fatalf("UnhealthyCount = %d, want 1", result.Diagnostics.UnhealthyCount)
		}
	})
}

func TestSelectBackendCandidateCountsUnhealthyByCandidate(t *testing.T) {
	base := time.Unix(600, 0)

	t.Run("mixed unhealthy replicas and missing telemetry reports missing", func(t *testing.T) {
		store := newStoreWithClock(5*time.Second, func() time.Time { return base })
		for _, telemetry := range []BackendTelemetry{
			{
				Identity:    BackendIdentity{BackendID: "backend-a", ModelName: "qwen3-8b", ReplicaID: "a-0"},
				Health:      HealthStateUnhealthy,
				CollectedAt: base,
			},
			{
				Identity:    BackendIdentity{BackendID: "backend-a", ModelName: "qwen3-8b", ReplicaID: "a-1"},
				Health:      HealthStateUnhealthy,
				CollectedAt: base,
			},
		} {
			if err := store.Upsert(telemetry); err != nil {
				t.Fatalf("Upsert returned error: %v", err)
			}
		}

		result := SelectBackendCandidate("qwen3-8b", []BackendCandidate{
			{BackendID: "backend-a"},
			{BackendID: "backend-b"},
		}, store)
		assertFailOpenReason(t, result, FallbackReasonMissingTelemetry)
		if result.Diagnostics.UnhealthyCount != 1 {
			t.Fatalf("UnhealthyCount = %d, want 1 unhealthy candidate", result.Diagnostics.UnhealthyCount)
		}
		if result.Diagnostics.FreshCandidateCount != 1 {
			t.Fatalf("FreshCandidateCount = %d, want 1", result.Diagnostics.FreshCandidateCount)
		}
	})

	t.Run("all unhealthy counts candidates not replicas", func(t *testing.T) {
		store := newStoreWithClock(5*time.Second, func() time.Time { return base })
		for _, telemetry := range []BackendTelemetry{
			{
				Identity:    BackendIdentity{BackendID: "backend-a", ModelName: "qwen3-8b", ReplicaID: "a-0"},
				Health:      HealthStateUnhealthy,
				CollectedAt: base,
			},
			{
				Identity:    BackendIdentity{BackendID: "backend-a", ModelName: "qwen3-8b", ReplicaID: "a-1"},
				Health:      HealthStateUnhealthy,
				CollectedAt: base,
			},
			{
				Identity:    BackendIdentity{BackendID: "backend-b", ModelName: "qwen3-8b", ReplicaID: "b-0"},
				Health:      HealthStateUnhealthy,
				CollectedAt: base,
			},
		} {
			if err := store.Upsert(telemetry); err != nil {
				t.Fatalf("Upsert returned error: %v", err)
			}
		}

		result := SelectBackendCandidate("qwen3-8b", []BackendCandidate{
			{BackendID: "backend-a"},
			{BackendID: "backend-b"},
		}, store)
		assertFailOpenReason(t, result, FallbackReasonAllUnhealthy)
		if result.Diagnostics.UnhealthyCount != 2 {
			t.Fatalf("UnhealthyCount = %d, want 2 unhealthy candidates", result.Diagnostics.UnhealthyCount)
		}
		if result.Diagnostics.FreshCandidateCount != 2 {
			t.Fatalf("FreshCandidateCount = %d, want 2", result.Diagnostics.FreshCandidateCount)
		}
	})
}

func TestSelectBackendCandidateSelectsBackendWithAnyHealthyReplica(t *testing.T) {
	base := time.Unix(700, 0)
	store := newStoreWithClock(5*time.Second, func() time.Time { return base })
	queueUnhealthy := 0
	queueHealthy := 5
	queueOther := 1
	for _, telemetry := range []BackendTelemetry{
		{
			Identity:    BackendIdentity{BackendID: "backend-a", ModelName: "qwen3-8b", ReplicaID: "a-0"},
			QueueDepth:  &queueUnhealthy,
			Health:      HealthStateUnhealthy,
			CollectedAt: base,
		},
		{
			Identity:    BackendIdentity{BackendID: "backend-a", ModelName: "qwen3-8b", ReplicaID: "a-1"},
			QueueDepth:  &queueHealthy,
			Health:      HealthStateHealthy,
			CollectedAt: base,
		},
		{
			Identity:    BackendIdentity{BackendID: "backend-b", ModelName: "qwen3-8b", ReplicaID: "b-0"},
			QueueDepth:  &queueOther,
			Health:      HealthStateUnhealthy,
			CollectedAt: base,
		},
	} {
		if err := store.Upsert(telemetry); err != nil {
			t.Fatalf("Upsert returned error: %v", err)
		}
	}

	result := SelectBackendCandidate("qwen3-8b", []BackendCandidate{
		{BackendID: "backend-a"},
		{BackendID: "backend-b"},
	}, store)
	if result.FailOpen {
		t.Fatalf("expected selectable healthy replica, got fail-open result %#v", result)
	}
	if result.SelectedBackendID != "backend-a" || result.SelectedReplicaID != "a-1" {
		t.Fatalf("expected backend-a/a-1, got %#v", result)
	}
	if result.Diagnostics.UnhealthyCount != 1 {
		t.Fatalf("UnhealthyCount = %d, want 1 unhealthy candidate", result.Diagnostics.UnhealthyCount)
	}
	if result.Diagnostics.FreshCandidateCount != 2 {
		t.Fatalf("FreshCandidateCount = %d, want 2", result.Diagnostics.FreshCandidateCount)
	}
}

func assertFailOpenReason(t *testing.T, result BackendPolicyResult, reason string) {
	t.Helper()
	if !result.FailOpen {
		t.Fatalf("expected fail-open result for %s, got %#v", reason, result)
	}
	if result.Reason != reason || result.Diagnostics.FallbackReason != reason {
		t.Fatalf("expected reason %q, got %#v", reason, result)
	}
	if result.SelectedBackendID != "" || result.SelectedReplicaID != "" {
		t.Fatalf("expected no selected backend on fail-open, got %#v", result)
	}
}
