package modelservice

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"go.uber.org/zap/zaptest/observer"
)

func TestSupervisorLogsAProcessKilledAfterTheGracePeriod(t *testing.T) {
	core, logs := observer.New(zapcore.WarnLevel)
	t.Cleanup(zap.ReplaceGlobals(zap.New(core)))
	grace := stopGracePeriod
	stopGracePeriod = 100 * time.Millisecond
	t.Cleanup(func() { stopGracePeriod = grace })
	// The shell creates ready once it ignores SIGTERM; a SIGTERM before that
	// ends it within the grace period, and nothing is killed.
	ready := filepath.Join(t.TempDir(), "ready")
	s := &supervisor{process: "runtime-0", deployments: []string{"kai"}, command: []string{"sh", "-c", `trap '' TERM; : > "$1"; sleep 30`, "sh", ready}}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		s.run(ctx)
		close(done)
	}()
	for deadline := time.Now().Add(5 * time.Second); ; time.Sleep(10 * time.Millisecond) {
		if _, err := os.Stat(ready); err == nil {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("the process never started ignoring SIGTERM")
		}
	}
	stopping := time.Now()
	cancel()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("a process that ignores SIGTERM must be killed after the grace period")
	}
	killed := logs.FilterMessage("runtime_process_killed").All()
	if len(killed) != 1 {
		t.Fatalf("a forced stop must be logged once: %+v", logs.All())
	}
	if fields := killed[0].ContextMap(); fields["process"] != "runtime-0" || fields["grace_period"] != "100ms" {
		t.Fatalf("the log must name the process and the grace period: %+v", fields)
	}
	if stopped := time.Since(stopping); stopped < stopGracePeriod {
		t.Fatalf("the process was killed %v after the stop, within its %v grace period", stopped, stopGracePeriod)
	}
}
