package workflowstore

import (
	"errors"
	"path/filepath"
	"strconv"
	"sync"
	"testing"
	"time"
)

func TestMLJobSurvivesStoreReopen(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	path := filepath.Join(dir, "wf.sqlite")

	s1, err := Open(path, Options{})
	if err != nil {
		t.Fatal(err)
	}
	now := time.Now().UTC().Truncate(time.Millisecond)
	job := MLJobRecord{
		ID:          "ml-benchmark-1",
		Type:        "benchmark",
		Status:      "running",
		CreatedAt:   now,
		Progress:    42,
		CurrentStep: "Running benchmark",
	}
	if putErr := s1.PutMLJob(job); putErr != nil {
		t.Fatal(putErr)
	}
	if appendErr := s1.AppendMLProgressEvent(job.ID, job.CurrentStep, job.Progress, "typed progress"); appendErr != nil {
		t.Fatal(appendErr)
	}
	_ = s1.Close()

	s2, err := Open(path, Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer s2.Close()

	got, err := s2.GetMLJob(job.ID)
	if err != nil || got == nil {
		t.Fatalf("GetMLJob: %v, %v", got, err)
	}
	if got.Status != "running" || got.Progress != 42 {
		t.Fatalf("unexpected job: %+v", got)
	}
	evs, err := s2.ListMLProgressEvents(job.ID, 10)
	if err != nil || len(evs) != 1 {
		t.Fatalf("events: %v, %v", evs, err)
	}
	if evs[0].Message != "typed progress" {
		t.Fatalf("event: %+v", evs[0])
	}
}

func TestRecoverInterruptedMLJobs(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	s, err := Open(filepath.Join(dir, "wf.sqlite"), Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	running := MLJobRecord{
		ID: "ml-train-9", Type: "train", Status: "running",
		CreatedAt: time.Now().UTC(), Progress: 37, CurrentStep: "training",
	}
	pending := MLJobRecord{
		ID: "ml-config-2", Type: "config", Status: "pending",
		CreatedAt: time.Now().UTC(), Progress: 0,
	}
	completed := MLJobRecord{
		ID: "ml-benchmark-1", Type: "benchmark", Status: "completed",
		CreatedAt: time.Now().UTC(), CompletedAt: time.Now().UTC(), Progress: 100,
	}
	for _, job := range []MLJobRecord{running, pending, completed} {
		if putErr := s.PutMLJob(job); putErr != nil {
			t.Fatal(putErr)
		}
	}
	if recoverErr := s.RecoverInterruptedMLJobs("restart"); recoverErr != nil {
		t.Fatal(recoverErr)
	}
	gotRunning, _ := s.GetMLJob(running.ID)
	if gotRunning == nil || gotRunning.Status != "failed" || gotRunning.Error != "restart" || gotRunning.CurrentStep != mlPipelineRecoveredStep {
		t.Fatalf("running got %+v", gotRunning)
	}
	gotPending, _ := s.GetMLJob(pending.ID)
	if gotPending == nil || gotPending.Status != "failed" || gotPending.Error != "restart" || gotPending.CurrentStep != mlPipelineRecoveredStep {
		t.Fatalf("pending got %+v", gotPending)
	}
	gotCompleted, _ := s.GetMLJob(completed.ID)
	if gotCompleted == nil || gotCompleted.Status != "completed" || gotCompleted.Error != "" {
		t.Fatalf("completed got %+v", gotCompleted)
	}

	events, err := s.ListMLProgressEvents(running.ID, 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(events) != 1 || events[0].Step != mlPipelineRecoveredStep || events[0].Percent != running.Progress || events[0].Message != "restart" {
		t.Fatalf("running events %+v", events)
	}
}

func TestOpenClawMessageAppendIncremental(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	s, err := Open(filepath.Join(dir, "wf.sqlite"), Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	room := "team-alpha"
	m1 := `{"id":"m1","roomId":"` + room + `","content":"a"}`
	m2 := `{"id":"m2","roomId":"` + room + `","content":"b"}`
	if appendErr := s.AppendOpenClawRoomMessage(room, "m1", m1); appendErr != nil {
		t.Fatal(appendErr)
	}
	if appendErr := s.AppendOpenClawRoomMessage(room, "m2", m2); appendErr != nil {
		t.Fatal(appendErr)
	}
	lines, err := s.ListOpenClawRoomMessages(room)
	if err != nil || len(lines) != 2 {
		t.Fatalf("messages: %v, %v", lines, err)
	}
	if lines[0] != m1 || lines[1] != m2 {
		t.Fatalf("order/content: %v", lines)
	}
}

func TestOpenClawMessageBoundedRecentAndUpdate(t *testing.T) {
	t.Parallel()
	s, err := Open(filepath.Join(t.TempDir(), "wf.sqlite"), Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	for index := 1; index <= 3; index++ {
		id := "m" + strconv.Itoa(index)
		payload := `{"id":"` + id + `"}`
		if appendErr := s.AppendOpenClawRoomMessageBounded("room-a", id, payload, 3); appendErr != nil {
			t.Fatalf("append %s: %v", id, appendErr)
		}
	}
	if appendErr := s.AppendOpenClawRoomMessageBounded("room-a", "m4", `{"id":"m4"}`, 3); !errors.Is(appendErr, ErrOpenClawRoomMessageLimit) {
		t.Fatalf("fourth append error = %v, want message limit", appendErr)
	}

	recent, err := s.ListRecentOpenClawRoomMessages("room-a", 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(recent) != 2 || recent[0] != `{"id":"m2"}` || recent[1] != `{"id":"m3"}` {
		t.Fatalf("unexpected recent chronological window: %v", recent)
	}

	if updateErr := s.UpdateOpenClawRoomMessageJSON("room-a", "m2", `{"id":"m2","updated":true}`); updateErr != nil {
		t.Fatal(updateErr)
	}
	payload, found, err := s.GetOpenClawRoomMessageJSON("room-a", "m2")
	if err != nil || !found || payload != `{"id":"m2","updated":true}` {
		t.Fatalf("updated payload = %q, %v, %v", payload, found, err)
	}
}

func TestOpenClawMessageBoundedConcurrentWritersDoNotOvershoot(t *testing.T) {
	t.Parallel()
	s, err := Open(filepath.Join(t.TempDir(), "wf.sqlite"), Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	const maximum = 10
	var wg sync.WaitGroup
	errorsSeen := make(chan error, 64)
	for index := 0; index < 64; index++ {
		wg.Add(1)
		go func(index int) {
			defer wg.Done()
			id := "m" + strconv.Itoa(index)
			errorsSeen <- s.AppendOpenClawRoomMessageBounded(
				"room-a",
				id,
				`{"id":"`+id+`"}`,
				maximum,
			)
		}(index)
	}
	wg.Wait()
	close(errorsSeen)
	for appendErr := range errorsSeen {
		if appendErr != nil && !errors.Is(appendErr, ErrOpenClawRoomMessageLimit) {
			t.Fatalf("unexpected append error: %v", appendErr)
		}
	}
	messages, err := s.ListOpenClawRoomMessages("room-a")
	if err != nil {
		t.Fatal(err)
	}
	if len(messages) != maximum {
		t.Fatalf("stored messages = %d, want %d", len(messages), maximum)
	}
}
