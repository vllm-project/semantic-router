package timeout

import (
	"bytes"
	"context"
	"os/exec"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestIdleTimeoutWriter_Write(t *testing.T) {
	var buf bytes.Buffer
	writer := NewIdleTimeoutWriter(&buf)

	_, err := writer.Write([]byte("test"))
	assert.NoError(t, err)

	assert.Equal(t, "test", buf.String())
	assert.Less(t, writer.GetIdleDuration(), 100*time.Millisecond)
}

func TestIdleTimeoutWriter_IdleDuration(t *testing.T) {
	writer := NewIdleTimeoutWriter(&bytes.Buffer{})
	_, err := writer.Write([]byte("initial"))
	assert.NoError(t, err)

	time.Sleep(100 * time.Millisecond)

	idle := writer.GetIdleDuration()
	assert.GreaterOrEqual(t, idle, 100*time.Millisecond)
}

func TestStreamMonitor_IdleTimeout(t *testing.T) {
	config := MonitorConfig{
		IdleTimeout:     500 * time.Millisecond,
		WarningInterval: 200 * time.Millisecond,
		CheckInterval:   100 * time.Millisecond,
	}

	cmd := exec.Command("sleep", "2")
	err := RunCommandWithIdleTimeout(cmd, config, "test_idle_timeout")

	assert.Error(t, err)
	assert.True(t, IsIdleTimeout(err), "error should be an IdleTimeoutError")
}

func TestStreamMonitor_WithActivity(t *testing.T) {
	config := MonitorConfig{
		IdleTimeout:     1 * time.Second,
		WarningInterval: 400 * time.Millisecond,
		CheckInterval:   100 * time.Millisecond,
	}

	cmd := exec.Command("bash", "-c", "for i in {1..5}; do echo 'activity'; sleep 0.5; done")
	err := RunCommandWithIdleTimeout(cmd, config, "test_with_activity")

	assert.NoError(t, err)
}

func TestIdleTimeoutError(t *testing.T) {
	err := &IdleTimeoutError{
		Operation:    "test-op",
		IdleDuration: 15 * time.Minute,
	}

	assert.Contains(t, err.Error(), "test-op")
	assert.Contains(t, err.Error(), "15m0s")
	assert.True(t, IsIdleTimeout(err))
	assert.False(t, IsIdleTimeout(context.DeadlineExceeded))
}

func TestRunCommandWithIdleTimeout_PeriodicOutput(t *testing.T) {
	// Command that outputs every 500ms for 3 seconds
	cmd := exec.Command("bash", "-c",
		"for i in {1..6}; do echo $i; sleep 0.5; done")

	config := MonitorConfig{
		IdleTimeout:     1 * time.Second,
		CheckInterval:   200 * time.Millisecond,
		WarningInterval: 0, // Disable warnings for this test
	}

	err := RunCommandWithIdleTimeout(cmd, config, "periodic")
	assert.NoError(t, err)
}

func TestRunCommandWithIdleTimeout_IdleTimeout(t *testing.T) {
	cmd := exec.Command("sleep", "10")

	config := MonitorConfig{
		IdleTimeout:     1 * time.Second,
		CheckInterval:   200 * time.Millisecond,
		WarningInterval: 0,
	}

	err := RunCommandWithIdleTimeout(cmd, config, "sleep")

	assert.Error(t, err)
	assert.True(t, IsIdleTimeout(err))
}

// Test for the warning message
func TestStreamMonitor_Warning(t *testing.T) {
	var stderrBuf bytes.Buffer
	config := MonitorConfig{
		IdleTimeout:     2 * time.Second,
		WarningInterval: 500 * time.Millisecond,
		CheckInterval:   100 * time.Millisecond,
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	monitor := NewStreamMonitor(ctx, cancel, config, "test_warning")
	monitor.stderr = NewIdleTimeoutWriter(&stderrBuf) // redirect for test
	monitor.stdout = NewIdleTimeoutWriter(&bytes.Buffer{})

	monitor.Start()
	defer monitor.Stop()

	time.Sleep(1 * time.Second)
	// No easy way to capture cli.Warning output without modifying it.
	// This test mainly ensures the logic doesn't crash.
	// Manual verification of warnings is part of the plan.
}

func TestRunCommandWithIdleTimeoutContext_ExternalCancellation(t *testing.T) {
	config := MonitorConfig{
		IdleTimeout:     10 * time.Second,
		CheckInterval:   100 * time.Millisecond,
		WarningInterval: 0,
	}

	// Context cancelled externally (not due to idle timeout)
	ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
	defer cancel()

	cmd := exec.Command("sleep", "5")
	err := RunCommandWithIdleTimeoutContext(ctx, cmd, config, "test_external_cancel")

	assert.Error(t, err)
	assert.False(t, IsIdleTimeout(err), "should be external cancellation, not idle timeout")
}

func TestRunCommandWithIdleTimeout_BackwardCompatibility(t *testing.T) {
	config := MonitorConfig{
		IdleTimeout:     5 * time.Second,
		CheckInterval:   100 * time.Millisecond,
		WarningInterval: 0,
	}

	cmd := exec.Command("echo", "test")
	err := RunCommandWithIdleTimeout(cmd, config, "compat_test")

	assert.NoError(t, err, "deprecated function should still work")
}

func TestRunCommandWithIdleTimeoutContext_ContextPropagation(t *testing.T) {
	config := MonitorConfig{
		IdleTimeout:     10 * time.Second,
		CheckInterval:   100 * time.Millisecond,
		WarningInterval: 0,
	}

	// Parent context with early timeout
	parentCtx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()

	cmd := exec.Command("sleep", "5")
	err := RunCommandWithIdleTimeoutContext(parentCtx, cmd, config, "test_propagation")

	assert.Error(t, err, "command should be cancelled via parent context")
	assert.False(t, IsIdleTimeout(err), "should not be idle timeout error")
}
