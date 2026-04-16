package workflowstore

import (
	"path/filepath"
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

	j := MLJobRecord{
		ID: "ml-train-9", Type: "train", Status: "running",
		CreatedAt: time.Now().UTC(),
	}
	if err := s.PutMLJob(j); err != nil {
		t.Fatal(err)
	}
	if err := s.RecoverInterruptedMLJobs("restart"); err != nil {
		t.Fatal(err)
	}
	got, _ := s.GetMLJob(j.ID)
	if got == nil || got.Status != "failed" || got.Error != "restart" {
		t.Fatalf("got %+v", got)
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
