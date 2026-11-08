package training

import (
	"errors"
	"reflect"
	"strings"
	"testing"

	c "github.com/vllm-project/semantic-router/src/semantic-router/pkg/trainingcontract"
)

func TestRetrySettlesRetainedSuccessAfterCancellation(t *testing.T) {
	root := t.TempDir()
	s, store := openService(t, root)
	t.Cleanup(func() { _ = store.Close() })
	req := seed(t, s, "alice", fixture(t, "selector"))
	req.Spec.Tasks = req.Spec.Tasks[:1]
	g, err := s.Submit(t.Context(), "alice", req)
	if err != nil {
		t.Fatal(err)
	}
	work, err := s.StartAttempt(t.Context(), "alice", g.Run.ID, g.Tasks[0].ID)
	if err != nil {
		t.Fatal(err)
	}
	file, err := s.PutOutput(t.Context(), "alice", g.Run.ID, work.AttemptID, strings.NewReader("model"))
	if err != nil {
		t.Fatal(err)
	}
	if _, operationErr := s.Cancel(t.Context(), "alice", g.Run.ID); operationErr != nil {
		t.Fatal(operationErr)
	}
	result := c.WorkerResult{SchemaVersion: c.Version, Status: c.Succeeded, Artifacts: []c.ArtifactResult{{Profile: work.Snapshot.Profile, Variants: []c.ArtifactVariantSpec{{Format: c.Component{Name: "mock", Version: "v1"}, Files: map[string]c.File{"model.bin": file}}}}}}
	cancelled, err := s.Complete(t.Context(), "alice", g.Run.ID, work.AttemptID, result)
	if err != nil || cancelled.Run.Status != c.Cancelled || cancelled.Tasks[0].Status != c.Succeeded || len(cancelled.Outputs.ArtifactIDs) != 1 {
		t.Fatalf("late success=%+v err=%v", cancelled, err)
	}
	before, err := s.Events(t.Context(), "alice", g.Run.ID, 0)
	if err != nil {
		t.Fatal(err)
	}
	retried, err := s.Retry(t.Context(), "alice", g.Run.ID)
	if err != nil || retried.Run.Status != c.Succeeded {
		t.Fatalf("retry with no remaining work=%+v err=%v", retried, err)
	}
	if retried.Run.Metadata != cancelled.Run.Metadata || !reflect.DeepEqual(retried.Run.Spec, cancelled.Run.Spec) || !reflect.DeepEqual(retried.Tasks, cancelled.Tasks) || !reflect.DeepEqual(retried.Outputs, cancelled.Outputs) {
		t.Fatal("retry changed frozen inputs, task/attempt history or output identities")
	}
	page, err := s.Events(t.Context(), "alice", g.Run.ID, before.NextAfter)
	if err != nil {
		t.Fatal(err)
	}
	status := cancelled.Run.Status
	for _, event := range page.Events {
		if event.TaskID != "" {
			t.Fatalf("retry unexpectedly changed task state: %+v", event)
		}
		if operationErr := c.ValidateTransition(status, event.Status); operationErr != nil {
			t.Fatalf("retry emitted invalid transition: %v", operationErr)
		}
		status = event.Status
	}
	if status != c.Succeeded {
		t.Fatalf("events did not reach success: %+v", page.Events)
	}
	if operationErr := store.Close(); operationErr != nil {
		t.Fatal(operationErr)
	}
	s, store = openService(t, root)
	if saved := graph(t, s, "alice", g.Run.ID); !reflect.DeepEqual(saved, retried) {
		t.Fatal("restart lost the settled retry")
	}
	if replayed, operationErr := s.Complete(t.Context(), "alice", g.Run.ID, work.AttemptID, result); operationErr != nil || !reflect.DeepEqual(replayed, retried) {
		t.Fatalf("late result replay changed retained outputs: %+v %v", replayed, operationErr)
	}
	if _, operationErr := s.Retry(t.Context(), "alice", g.Run.ID); !errors.Is(operationErr, ErrConflict) {
		t.Fatalf("settled run accepted another retry: %v", operationErr)
	}
}
