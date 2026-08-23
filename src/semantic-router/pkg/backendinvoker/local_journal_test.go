package backendinvoker

import (
	"context"
	"testing"
	"time"
)

func TestProcessLocalJournalValidatesInvokerOwnedEvidence(t *testing.T) {
	journal := ProcessLocalJournal{}
	plan := testPlan()
	if err := journal.BeginDispatch(context.Background(), plan, time.Now().Add(time.Minute)); err != nil {
		t.Fatal(err)
	}
	attempt := Attempt{ID: "dsp:1", Number: 1, BackendID: "be", StartedAt: time.Now().UTC()}
	if err := journal.BeginAttempt(context.Background(), plan, attempt); err != nil {
		t.Fatal(err)
	}
	if err := journal.FinishAttempt(context.Background(), plan, AttemptResult{
		Attempt: attempt, State: AttemptKnownZero, CompletedAt: attempt.StartedAt.Add(time.Millisecond),
	}); err != nil {
		t.Fatal(err)
	}

	attempt.BackendID = "another-backend"
	if err := journal.BeginAttempt(context.Background(), plan, attempt); err == nil {
		t.Fatal("unknown backend was accepted")
	}
}

func TestProcessLocalJournalRejectsExpiredDispatch(t *testing.T) {
	if err := (ProcessLocalJournal{}).BeginDispatch(
		context.Background(), testPlan(), time.Now().Add(-time.Second),
	); err == nil {
		t.Fatal("expired local dispatch was accepted")
	}
}
