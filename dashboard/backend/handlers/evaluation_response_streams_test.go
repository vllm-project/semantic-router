package handlers

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/evaluationplane"
)

func TestEvaluationResponseStreamLimiterPreservesPrincipalFairness(t *testing.T) {
	limiter := newEvaluationResponseStreamLimiter()
	ownerA, err := evaluationplane.NewActor("stream-owner-a", false)
	if err != nil {
		t.Fatal(err)
	}
	ownerB, err := evaluationplane.NewActor("stream-owner-b", false)
	if err != nil {
		t.Fatal(err)
	}
	releases := make([]func(), 0, maxEvaluationResponseStreamsPerPrincipal)
	for range maxEvaluationResponseStreamsPerPrincipal {
		release, acquireErr := limiter.acquire(ownerA)
		if acquireErr != nil {
			t.Fatalf("owner A stream below principal bound: %v", acquireErr)
		}
		releases = append(releases, release)
	}
	if release, acquireErr := limiter.acquire(ownerA); !errors.Is(acquireErr, evaluationplane.ErrConflict) || release != nil {
		t.Fatalf("owner A exceeded principal bound release=%v err=%v", release != nil, acquireErr)
	}
	releaseB, acquireErr := limiter.acquire(ownerB)
	if acquireErr != nil {
		t.Fatalf("owner A starvation leaked across principals: %v", acquireErr)
	}
	releaseB()
	for _, release := range releases {
		release()
		release() // release is idempotent and must not underflow accounting.
	}
	if limiter.total != 0 || len(limiter.byPrincipal) != 0 {
		t.Fatalf("stream limiter leaked total=%d principals=%d", limiter.total, len(limiter.byPrincipal))
	}
}

type evaluationDeadlineProbe struct {
	header       http.Header
	deadlines    []time.Time
	writes       int
	flushes      int
	writeFailure error
}

func (probe *evaluationDeadlineProbe) Header() http.Header {
	if probe.header == nil {
		probe.header = make(http.Header)
	}
	return probe.header
}

func (probe *evaluationDeadlineProbe) WriteHeader(int) {}

func (probe *evaluationDeadlineProbe) Write(data []byte) (int, error) {
	probe.writes++
	if probe.writeFailure != nil {
		return 0, probe.writeFailure
	}
	return len(data), nil
}

func (probe *evaluationDeadlineProbe) SetWriteDeadline(deadline time.Time) error {
	probe.deadlines = append(probe.deadlines, deadline)
	return nil
}

func (probe *evaluationDeadlineProbe) FlushError() error {
	probe.flushes++
	return nil
}

func (probe *evaluationDeadlineProbe) Flush() { probe.flushes++ }

func TestEvaluationDeadlineWriterArmsEveryWriteAndFlush(t *testing.T) {
	probe := &evaluationDeadlineProbe{}
	writer := newEvaluationDeadlineWriter(probe, time.Second)
	if _, err := writer.Write([]byte("evidence")); err != nil {
		t.Fatal(err)
	}
	if err := writer.flush(); err != nil {
		t.Fatal(err)
	}
	writer.clearDeadline()
	if probe.writes != 1 || probe.flushes != 1 || len(probe.deadlines) != 3 ||
		probe.deadlines[0].IsZero() || probe.deadlines[1].IsZero() || !probe.deadlines[2].IsZero() {
		t.Fatalf("deadline contract writes=%d flushes=%d deadlines=%v", probe.writes, probe.flushes, probe.deadlines)
	}

	probe.writeFailure = os.ErrDeadlineExceeded
	if _, err := writer.Write([]byte("slow client")); !errors.Is(err, os.ErrDeadlineExceeded) {
		t.Fatalf("slow client write error=%v, want deadline exceeded", err)
	}
}

func TestEvaluationSSEWriteTimeoutReleasesSubscriberAndStreamQuota(t *testing.T) {
	service := newEvaluationHandlerService(t, "")
	run, err := service.CreateRunAs(context.Background(), evaluationplane.SystemActor(), evaluationplane.CreateRunRequest{
		ClientRequestID: "66c4d6c8-f4f5-423f-a8a6-31bd07702a28",
		Name:            "slow SSE client", SuiteIDs: []string{"evaluation-smoke"},
		TrackIDs: []evaluationplane.TrackID{"routing"}, Mode: evaluationplane.ModeReplay,
		TargetID: "fixture", ChangeProfile: "schema_adapter", SampleLimit: 4, Concurrency: 1, Seed: 17,
	})
	if err != nil {
		t.Fatalf("create SSE fixture: %v", err)
	}
	handler := newAuthenticatedEvaluationTestHandler(service, false)
	probe := &evaluationDeadlineProbe{writeFailure: os.ErrDeadlineExceeded}
	request := httptest.NewRequest(http.MethodGet, evaluationAPIBase+"/runs/"+run.ID+"/events", nil)
	handler.RunRoute(probe, request)
	if handler.responseStreams.total != 0 || len(handler.responseStreams.byPrincipal) != 0 {
		t.Fatalf("timed-out SSE leaked transport capacity: %+v", handler.responseStreams)
	}

	unsubscribers := make([]func(), 0, 16)
	for index := 0; index < 16; index++ {
		_, unsubscribe, subscribeErr := service.SubscribeAs(evaluationplane.SystemActor(), run.ID)
		if subscribeErr != nil {
			t.Fatalf("timed-out SSE leaked run subscriber %d: %v", index, subscribeErr)
		}
		unsubscribers = append(unsubscribers, unsubscribe)
	}
	for _, unsubscribe := range unsubscribers {
		unsubscribe()
	}
}
