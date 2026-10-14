package modelservice

import (
	"bufio"
	"context"
	"errors"
	"io"
	"os"
	"os/exec"
	"sync"
	"syscall"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

const (
	minRestartBackoff = time.Second
	maxRestartBackoff = time.Minute
	// A process that ran this long before exiting resets the back-off.
	stableRunDuration = 2 * time.Minute
)

// stopGracePeriod is how long a stopping runtime gets between SIGTERM and SIGKILL.
var stopGracePeriod = 10 * time.Second

// errRecycled marks the exit of a process the group recycled.
var errRecycled = errors.New("recycled after every model failed to load")

// supervisor runs one Router-managed runtime process and restarts it with
// exponential back-off until its context is cancelled: after it exits, and
// after the group recycles a process whose every model failed to load. onExit
// reports every exit (or failure to start) with how long the process ran.
type supervisor struct {
	process     string
	deployments []string
	command     []string
	env         []string
	socket      string
	onExit      func(err error, ran time.Duration)

	mu       sync.Mutex
	running  *os.Process
	recycled *os.Process
}

func (s *supervisor) run(ctx context.Context) {
	backoff := minRestartBackoff
	for attempt := 0; ; attempt++ {
		if attempt > 0 {
			for _, deployment := range s.deployments {
				restartsTotal.WithLabelValues(deployment).Inc()
			}
			select {
			case <-ctx.Done():
				return
			case <-time.After(backoff):
			}
		}
		started := time.Now()
		err := s.runOnce(ctx)
		if ctx.Err() != nil {
			return
		}
		ran := time.Since(started)
		if s.onExit != nil {
			s.onExit(err, ran)
		}
		fields := map[string]interface{}{"process": s.process, "deployments": s.deployments, "restart_in": backoff.String()}
		if err != nil {
			fields["error"] = err.Error()
		}
		logging.ComponentWarnEvent("model_runtime", "runtime_process_exited", fields)
		if ran > stableRunDuration {
			backoff = minRestartBackoff
		} else {
			backoff = min(backoff*2, maxRestartBackoff)
		}
	}
}

func (s *supervisor) runOnce(ctx context.Context) error {
	if len(s.command) == 0 {
		return errors.New("no runtime command")
	}
	path, err := exec.LookPath(s.command[0])
	if err != nil {
		return err
	}
	if staleErr := removeStaleSocket(s.socket); staleErr != nil {
		return staleErr
	}
	cmd := exec.Command(path, s.command[1:]...) //nolint:gosec // the command comes from Router configuration and environment
	cmd.Env = s.env
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}
	cmd.Stderr = cmd.Stdout
	if err := cmd.Start(); err != nil {
		return err
	}
	s.mu.Lock()
	s.running = cmd.Process
	s.mu.Unlock()
	defer func() {
		s.mu.Lock()
		s.running = nil
		s.mu.Unlock()
	}()
	logging.ComponentEvent("model_runtime", "runtime_process_started", map[string]interface{}{
		"process": s.process, "deployments": s.deployments, "pid": cmd.Process.Pid,
	})
	go s.forwardLogs(stdout)
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	select {
	case err := <-done:
		s.mu.Lock()
		recycled := s.recycled == cmd.Process
		s.mu.Unlock()
		if recycled {
			return errRecycled
		}
		return err
	case <-ctx.Done():
		s.terminate(done)
		return ctx.Err()
	}
}

// terminate stops the process group: SIGTERM, then SIGKILL after the grace period.
func (s *supervisor) terminate(done <-chan error) {
	s.mu.Lock()
	process := s.running
	s.mu.Unlock()
	if process == nil {
		return
	}
	_ = syscall.Kill(-process.Pid, syscall.SIGTERM)
	select {
	case <-done:
	case <-time.After(stopGracePeriod):
		s.kill(process)
		<-done
	}
}

// recycle stops the running process as an exit would, so run starts a new one
// after its back-off: SIGTERM, then SIGKILL after the grace period. It acts
// once per process and reports whether it did.
func (s *supervisor) recycle() bool {
	s.mu.Lock()
	process := s.running
	if process == nil || process == s.recycled {
		s.mu.Unlock()
		return false
	}
	s.recycled = process
	s.mu.Unlock()
	_ = syscall.Kill(-process.Pid, syscall.SIGTERM)
	time.AfterFunc(stopGracePeriod, func() {
		s.mu.Lock()
		running := s.running == process
		s.mu.Unlock()
		if running {
			s.kill(process)
		}
	})
	return true
}

// kill ends a process group that outlived its grace period, and records it.
func (s *supervisor) kill(process *os.Process) {
	logging.ComponentWarnEvent("model_runtime", "runtime_process_killed", map[string]interface{}{
		"process": s.process, "deployments": s.deployments, "pid": process.Pid, "grace_period": stopGracePeriod.String(),
	})
	_ = syscall.Kill(-process.Pid, syscall.SIGKILL)
}

func (s *supervisor) forwardLogs(reader io.Reader) {
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 64*1024), 1024*1024)
	for scanner.Scan() {
		logging.ComponentEvent("model_runtime", "runtime_log", map[string]interface{}{
			"process": s.process, "line": scanner.Text(),
		})
	}
}

// removeStaleSocket deletes a socket file left by a runtime that did not exit
// cleanly; any other file at that path is an error.
func removeStaleSocket(path string) error {
	if path == "" {
		return nil
	}
	info, err := os.Lstat(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	if info.Mode()&os.ModeSocket == 0 {
		return errors.New(path + " exists and is not a socket")
	}
	return os.Remove(path)
}
