package routerruntime

import (
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/configsnapshot"
)

// A reader that sees a finished attempt must also see what the attempt
// listeners recorded for it, such as its audit entry (#5031).
func TestFinishedAttemptIsPublishedAfterItsListenersRan(t *testing.T) {
	registry := NewRegistry(nil)
	entered, release := make(chan struct{}), make(chan struct{})
	registry.OnConfigAttempt(func(configsnapshot.Attempt) {
		close(entered)
		<-release
	})
	started := time.Now().UTC()
	registry.ReportConfigAttempt(configsnapshot.Attempt{
		ID: 1, Hash: "a", Status: configsnapshot.AttemptPreparing, StartedAt: started,
	}, nil)
	finished := started.Add(time.Millisecond)
	done := make(chan struct{})
	go func() {
		defer close(done)
		registry.ReportConfigAttempt(configsnapshot.Attempt{
			ID: 1, Hash: "a", Status: configsnapshot.AttemptActive, Version: 1, StartedAt: started, FinishedAt: &finished,
		}, nil)
	}()
	<-entered
	if activation := registry.ConfigActivation(); activation.Status != string(configsnapshot.AttemptPreparing) {
		close(release)
		t.Fatalf("activation = %+v published before its listeners finished", activation)
	}
	close(release)
	<-done
	if activation := registry.ConfigActivation(); activation.Status != string(configsnapshot.AttemptActive) || activation.FinishedAt == nil {
		t.Fatalf("finished activation = %+v", activation)
	}
}
