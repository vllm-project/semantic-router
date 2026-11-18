package modelservice

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"testing/iotest"
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

func TestSupervisorKeepsReadingAfterAnOversizedLine(t *testing.T) {
	// The output after the oversized line fills the pipe, so the shell creates
	// written only if the supervisor keeps reading.
	written := filepath.Join(t.TempDir(), "written")
	script := `head -c 1200000 /dev/zero | tr '\0' a; echo; yes after | head -n 40000; : > "$1"; sleep 30`
	s := &supervisor{process: "runtime-0", command: []string{"sh", "-c", script, "sh", written}}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		s.run(ctx)
		close(done)
	}()
	defer func() {
		cancel()
		<-done
	}()
	for deadline := time.Now().Add(5 * time.Second); ; time.Sleep(10 * time.Millisecond) {
		if _, err := os.Stat(written); err == nil {
			return
		}
		if time.Now().After(deadline) {
			t.Fatal("the process blocked writing after an oversized line")
		}
	}
}

func TestSupervisorForwardsAnOversizedLineInPieces(t *testing.T) {
	core, logs := observer.New(zapcore.InfoLevel)
	t.Cleanup(zap.ReplaceGlobals(zap.New(core)))
	s := &supervisor{process: "runtime-split"}
	s.forwardLogs(strings.NewReader(strings.Repeat("a", maxLogLine+10) + "\nafter\n"))
	var lines []string
	var lengths []int
	for _, entry := range logs.FilterMessage("runtime_log").FilterField(zap.String("process", "runtime-split")).All() {
		line, _ := entry.ContextMap()["line"].(string)
		lines = append(lines, line)
		lengths = append(lengths, len(line))
	}
	if len(lines) != 3 || len(lines[0]) != maxLogLine || lines[1] != strings.Repeat("a", 10) || lines[2] != "after" {
		t.Fatalf("an oversized line must be forwarded in pieces, then the lines after it: line lengths %v", lengths)
	}
}

func TestSupervisorLogsAFailedLogRead(t *testing.T) {
	core, logs := observer.New(zapcore.WarnLevel)
	t.Cleanup(zap.ReplaceGlobals(zap.New(core)))
	s := &supervisor{process: "runtime-0", deployments: []string{"kai"}}
	closed, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	_ = w.Close()
	_ = closed.Close()
	s.forwardLogs(closed)
	if logs.Len() != 0 {
		t.Fatalf("a pipe closed by Wait must not log a read failure: %+v", logs.All())
	}
	s.forwardLogs(iotest.ErrReader(errors.New("read failed")))
	failed := logs.FilterMessage("runtime_log_read_failed").All()
	if len(failed) != 1 || failed[0].ContextMap()["error"] != "read failed" || failed[0].ContextMap()["process"] != "runtime-0" {
		t.Fatalf("a failed read must be logged with the process and the error: %+v", logs.All())
	}
}
