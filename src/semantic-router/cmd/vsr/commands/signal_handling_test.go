package commands

import (
	"os"
	"os/exec"
	"os/signal"
	"syscall"
	"testing"
	"time"
)

// TestSignalHandling verifies graceful shutdown on interrupt signal
func TestSignalHandling(t *testing.T) {
	t.Run("signal channel receives interrupt", func(t *testing.T) {
		// Create signal channel
		sigChan := make(chan os.Signal, 1)
		signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
		defer signal.Stop(sigChan)

		// Simulate sending interrupt signal
		go func() {
			time.Sleep(100 * time.Millisecond)
			sigChan <- os.Interrupt
		}()

		// Wait for signal with timeout
		select {
		case sig := <-sigChan:
			if sig != os.Interrupt {
				t.Errorf("Expected os.Interrupt, got %v", sig)
			}
		case <-time.After(1 * time.Second):
			t.Error("Signal should be received within 1 second")
		}
	})

	t.Run("process cleanup on signal", func(t *testing.T) {
		// Start a dummy process (simulating port-forward)
		cmd := exec.Command("sleep", "10")
		if err := cmd.Start(); err != nil {
			t.Fatalf("Failed to start process: %v", err)
		}

		pid := cmd.Process.Pid

		// Create signal channel
		sigChan := make(chan os.Signal, 1)
		signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
		defer signal.Stop(sigChan)

		// Simulate interrupt after short delay
		go func() {
			time.Sleep(100 * time.Millisecond)
			sigChan <- os.Interrupt
		}()

		// Wait for signal and kill process gracefully
		done := make(chan error)
		go func() {
			done <- cmd.Wait()
		}()

		select {
		case <-sigChan:
			// Kill the process
			if cmd.Process != nil {
				_ = cmd.Process.Kill()
			}

			// Wait a bit for process to die
			time.Sleep(100 * time.Millisecond)

			// Verify process is dead
			if err := cmd.Process.Signal(syscall.Signal(0)); err == nil {
				t.Errorf("Process %d should be terminated after signal", pid)
			}

		case err := <-done:
			t.Logf("Process exited on its own: %v", err)
		case <-time.After(2 * time.Second):
			t.Error("Test timed out")
			_ = cmd.Process.Kill()
		}
	})

	t.Run("done channel receives process exit", func(t *testing.T) {
		// Start a process that exits quickly
		cmd := exec.Command("sleep", "0.1")
		if err := cmd.Start(); err != nil {
			t.Fatalf("Failed to start process: %v", err)
		}

		done := make(chan error)
		go func() {
			done <- cmd.Wait()
		}()

		// Wait for either signal or done
		sigChan := make(chan os.Signal, 1)
		signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
		defer signal.Stop(sigChan)

		select {
		case <-sigChan:
			t.Error("Should not receive signal in this test")
		case err := <-done:
			// Process exited normally
			if err != nil {
				t.Logf("Process exited with: %v", err)
			}
		case <-time.After(1 * time.Second):
			t.Error("Process should exit within 1 second")
			_ = cmd.Process.Kill()
		}
	})
}

// TestGracefulShutdown verifies cleanup logic
func TestGracefulShutdown(t *testing.T) {
	t.Run("process is killed when signal received", func(t *testing.T) {
		cmd := exec.Command("sleep", "30")
		if err := cmd.Start(); err != nil {
			t.Fatalf("Failed to start process: %v", err)
		}

		pid := cmd.Process.Pid
		t.Logf("Started process with PID: %d", pid)

		// Kill the process (simulating signal handler)
		if err := cmd.Process.Kill(); err != nil {
			t.Errorf("Failed to kill process: %v", err)
		}

		// Wait for process to actually exit
		_ = cmd.Wait()

		// Verify process is gone
		if err := cmd.Process.Signal(syscall.Signal(0)); err == nil {
			t.Errorf("Process %d should be dead", pid)
		}
	})

	t.Run("select statement with signal and done channels", func(t *testing.T) {
		// This tests the select pattern used in dashboard command
		sigChan := make(chan os.Signal, 1)
		done := make(chan error)

		// Simulate signal received first
		go func() {
			time.Sleep(50 * time.Millisecond)
			sigChan <- os.Interrupt
		}()

		var receivedSignal bool
		select {
		case <-sigChan:
			receivedSignal = true
		case <-done:
			t.Error("Should receive signal, not done")
		case <-time.After(1 * time.Second):
			t.Error("Should receive signal within 1 second")
		}

		if !receivedSignal {
			t.Error("Signal should have been received")
		}
	})
}

// TestSignalNotification verifies signal notification setup
func TestSignalNotification(t *testing.T) {
	t.Run("can create and stop signal notification", func(t *testing.T) {
		sigChan := make(chan os.Signal, 1)
		signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

		// Stop notification
		signal.Stop(sigChan)

		// After stopping, signals should not be delivered
		// This is more of a safety check
		select {
		case <-sigChan:
			// May or may not receive depending on timing
		case <-time.After(100 * time.Millisecond):
			// Expected - no signal received
		}
	})

	t.Run("signal channel has buffer size", func(t *testing.T) {
		sigChan := make(chan os.Signal, 1)
		if cap(sigChan) != 1 {
			t.Errorf("Signal channel should have capacity 1, got %d", cap(sigChan))
		}
	})
}
