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

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

const (
	minRestartBackoff = time.Second
	maxRestartBackoff = time.Minute
	stopGracePeriod   = 10 * time.Second
	// A process that ran this long before exiting resets the back-off.
	stableRunDuration = 2 * time.Minute
)

// supervisor runs one Router-managed runtime process and restarts it with
// exponential back-off until its context is cancelled.
type supervisor struct {
	name    string
	command []string
	env     []string
	socket  string

	mu      sync.Mutex
	process *os.Process
}

func managedCommand(base []string, deployment config.ModelDeployment, socket, cacheDir string) []string {
	command := append(append([]string(nil), base...), "serve", deployment.Artifact,
		"--uds", socket,
		"--device", deployment.Device,
		"--profile", deployment.Profile,
	)
	if deployment.Revision != "" {
		command = append(command, "--revision", deployment.Revision)
	}
	if cacheDir != "" {
		command = append(command, "--cache-dir", cacheDir)
	}
	return command
}

func (s *supervisor) run(ctx context.Context) {
	backoff := minRestartBackoff
	for attempt := 0; ; attempt++ {
		if attempt > 0 {
			restartsTotal.WithLabelValues(s.name).Inc()
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
		fields := map[string]interface{}{"deployment": s.name, "restart_in": backoff.String()}
		if err != nil {
			fields["error"] = err.Error()
		}
		logging.ComponentWarnEvent("model_runtime", "runtime_process_exited", fields)
		if time.Since(started) > stableRunDuration {
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
	s.process = cmd.Process
	s.mu.Unlock()
	logging.ComponentEvent("model_runtime", "runtime_process_started", map[string]interface{}{
		"deployment": s.name, "pid": cmd.Process.Pid,
	})
	go s.forwardLogs(stdout)
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	select {
	case err := <-done:
		return err
	case <-ctx.Done():
		s.terminate(done)
		return ctx.Err()
	}
}

// terminate stops the process group: SIGTERM, then SIGKILL after the grace period.
func (s *supervisor) terminate(done <-chan error) {
	s.mu.Lock()
	process := s.process
	s.mu.Unlock()
	if process == nil {
		return
	}
	_ = syscall.Kill(-process.Pid, syscall.SIGTERM)
	select {
	case <-done:
	case <-time.After(stopGracePeriod):
		_ = syscall.Kill(-process.Pid, syscall.SIGKILL)
		<-done
	}
}

func (s *supervisor) forwardLogs(reader io.Reader) {
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 64*1024), 1024*1024)
	for scanner.Scan() {
		logging.ComponentEvent("model_runtime", "runtime_log", map[string]interface{}{
			"deployment": s.name, "line": scanner.Text(),
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
