package timeout

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"sync"
	"syscall"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cli"
)

// IdleTimeoutWriter wraps an io.Writer and tracks the last write time.
// It is thread-safe for concurrent writes.
type IdleTimeoutWriter struct {
	writer    io.Writer
	lastWrite time.Time
	mu        sync.Mutex
}

// NewIdleTimeoutWriter creates a new IdleTimeoutWriter.
func NewIdleTimeoutWriter(w io.Writer) *IdleTimeoutWriter {
	return &IdleTimeoutWriter{
		writer:    w,
		lastWrite: time.Now(),
	}
}

// Write writes to the underlying writer and updates the last write time.
func (w *IdleTimeoutWriter) Write(p []byte) (n int, err error) {
	w.mu.Lock()
	w.lastWrite = time.Now()
	w.mu.Unlock()
	if w.writer == nil {
		return len(p), nil
	}
	return w.writer.Write(p)
}

// GetIdleDuration returns the duration since the last write.
func (w *IdleTimeoutWriter) GetIdleDuration() time.Duration {
	w.mu.Lock()
	defer w.mu.Unlock()
	return time.Since(w.lastWrite)
}

// MonitorConfig holds the configuration for the idle timeout monitor.
type MonitorConfig struct {
	IdleTimeout     time.Duration // Max idle time before failure
	WarningInterval time.Duration // How often to show warnings
	CheckInterval   time.Duration // Monitoring check frequency
}

var (
	// DefaultConfig is the default configuration for the idle timeout monitor.
	DefaultConfig = MonitorConfig{
		IdleTimeout:     15 * time.Minute,
		WarningInterval: 5 * time.Minute,
		CheckInterval:   30 * time.Second,
	}
	// LongRunningConfig is an alias for the default config, used for clarity.
	LongRunningConfig = DefaultConfig
)

// StreamMonitor manages idle timeout monitoring for a command.
type StreamMonitor struct {
	stdout        *IdleTimeoutWriter
	stderr        *IdleTimeoutWriter
	config        MonitorConfig
	ctx           context.Context
	cancel        context.CancelFunc
	done          chan struct{}
	operationName string
	stopOnce      sync.Once
}

// NewStreamMonitor creates a new StreamMonitor.
func NewStreamMonitor(ctx context.Context, cancel context.CancelFunc, config MonitorConfig, operationName string) *StreamMonitor {
	return &StreamMonitor{
		stdout:        NewIdleTimeoutWriter(os.Stdout),
		stderr:        NewIdleTimeoutWriter(os.Stderr),
		config:        config,
		ctx:           ctx,
		cancel:        cancel,
		done:          make(chan struct{}),
		operationName: operationName,
	}
}

// GetStdout returns the wrapped stdout writer.
func (m *StreamMonitor) GetStdout() io.Writer {
	return m.stdout
}

// GetStderr returns the wrapped stderr writer.
func (m *StreamMonitor) GetStderr() io.Writer {
	return m.stderr
}

// Start begins the monitoring loop in a new goroutine.
func (m *StreamMonitor) Start() {
	go m.monitorLoop()
}

// Stop signals the monitoring goroutine to stop.
// Safe to call multiple times.
func (m *StreamMonitor) Stop() {
	m.stopOnce.Do(func() {
		close(m.done)
	})
}

// monitorLoop runs in a separate goroutine and monitors idle time.
// It accesses m.config fields which are read-only after StreamMonitor creation,
// making concurrent access safe without additional synchronization.
func (m *StreamMonitor) monitorLoop() {
	ticker := time.NewTicker(m.config.CheckInterval)
	defer ticker.Stop()

	var lastWarningIdleDuration time.Duration

	for {
		select {
		case <-ticker.C:
			idleDuration := min(m.stdout.GetIdleDuration(), m.stderr.GetIdleDuration())

			if idleDuration >= m.config.IdleTimeout {
				m.cancel()
				return
			}

			if m.config.WarningInterval > 0 {
				if idleDuration >= m.config.WarningInterval && (lastWarningIdleDuration == 0 || idleDuration-lastWarningIdleDuration >= m.config.WarningInterval) {
					cli.Warning(fmt.Sprintf(
						"No output for over %v while running '%s', still waiting...",
						idleDuration.Truncate(time.Second),
						m.operationName,
					))
					lastWarningIdleDuration = idleDuration
				}
			}

		case <-m.ctx.Done():
			return
		case <-m.done:
			return
		}
	}
}

// IdleTimeoutError indicates a command was terminated due to idle timeout.
type IdleTimeoutError struct {
	Operation    string
	IdleDuration time.Duration
}

func (e *IdleTimeoutError) Error() string {
	return fmt.Sprintf(
		"operation '%s' timed out after %v of inactivity (no output). "+
			"This may indicate the process is stuck or has encountered an issue. "+
			"Check the logs and try again.",
		e.Operation,
		e.IdleDuration,
	)
}

// IsIdleTimeout checks if an error is an IdleTimeoutError.
func IsIdleTimeout(err error) bool {
	var idleErr *IdleTimeoutError
	return errors.As(err, &idleErr)
}

// RunCommandWithIdleTimeoutContext executes a command with idle timeout monitoring
// using the provided parent context for cancellation propagation.
//
// The parent context allows external cancellation to propagate to the command,
// while still monitoring for idle timeouts independently.
func RunCommandWithIdleTimeoutContext(
	ctx context.Context,
	cmd *exec.Cmd,
	config MonitorConfig,
	operationName string,
) error {
	// Create cancellable context from parent (not Background)
	monitorCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	// We need to create a new command with the context.
	//nolint:gosec // G204: cmd.Path and cmd.Args are from a trusted source
	newCmd := exec.CommandContext(monitorCtx, cmd.Path, cmd.Args[1:]...)
	newCmd.Env = cmd.Env
	newCmd.Dir = cmd.Dir

	monitor := NewStreamMonitor(monitorCtx, cancel, config, operationName)

	newCmd.Stdout = monitor.GetStdout()
	newCmd.Stderr = monitor.GetStderr()

	newCmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	newCmd.Cancel = func() error {
		if newCmd.Process == nil {
			return nil
		}
		pgid, err := syscall.Getpgid(newCmd.Process.Pid)
		if err != nil {
			return newCmd.Process.Kill()
		}
		if err := syscall.Kill(-pgid, syscall.SIGTERM); err != nil {
			return newCmd.Process.Kill()
		}
		time.Sleep(2 * time.Second)
		return syscall.Kill(-pgid, syscall.SIGKILL)
	}

	monitor.Start()
	defer monitor.Stop()

	cmdErr := newCmd.Run()

	// Distinguish idle timeout from external cancellation
	if monitorCtx.Err() == context.Canceled {
		idleDuration := min(monitor.stdout.GetIdleDuration(), monitor.stderr.GetIdleDuration())
		if idleDuration >= config.IdleTimeout {
			return &IdleTimeoutError{
				Operation:    operationName,
				IdleDuration: config.IdleTimeout,
			}
		}
	}

	return cmdErr
}

// RunCommandWithIdleTimeout executes a command with idle timeout monitoring.
//
// Deprecated: Use RunCommandWithIdleTimeoutContext for proper context propagation.
// This function will be removed in a future version.
func RunCommandWithIdleTimeout(
	cmd *exec.Cmd,
	config MonitorConfig,
	operationName string,
) error {
	return RunCommandWithIdleTimeoutContext(context.Background(), cmd, config, operationName)
}
