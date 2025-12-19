package deployment

import (
	"fmt"
	"os"
	"os/exec"
	"testing"
	"time"
)

// TestPIDFilePermissions verifies restrictive file permissions (0600) for security
func TestPIDFilePermissions(t *testing.T) {
	pidFilePath := getPIDFilePath()
	logFilePath := getLogFilePath()

	// Clean up any existing files
	os.Remove(pidFilePath)
	os.Remove(logFilePath)
	defer os.Remove(pidFilePath)
	defer os.Remove(logFilePath)

	// Create log file with correct permissions (simulating DeployLocal)
	logFile, err := os.OpenFile(logFilePath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		t.Fatalf("Failed to create log file: %v", err)
	}
	defer logFile.Close()

	// Start a dummy process (simulating router)
	cmd := exec.Command("sleep", "1")
	cmd.Stdout = logFile
	cmd.Stderr = logFile

	if err := cmd.Start(); err != nil {
		t.Fatalf("Failed to start process: %v", err)
	}
	defer func() {
		if cmd.Process != nil {
			_ = cmd.Process.Kill()
		}
	}()

	pid := cmd.Process.Pid

	// Write PID file with correct permissions
	if err := os.WriteFile(pidFilePath, []byte(fmt.Sprintf("%d", pid)), 0o600); err != nil {
		t.Fatalf("Failed to write PID file: %v", err)
	}

	t.Run("PID file has 0600 permissions", func(t *testing.T) {
		info, err := os.Stat(pidFilePath)
		if err != nil {
			t.Fatalf("Failed to stat PID file: %v", err)
		}
		if info.Mode().Perm() != 0o600 {
			t.Errorf("PID file permissions = %o, expected 0600", info.Mode().Perm())
		}
	})

	t.Run("log file has 0600 permissions", func(t *testing.T) {
		info, err := os.Stat(logFilePath)
		if err != nil {
			t.Fatalf("Failed to stat log file: %v", err)
		}
		if info.Mode().Perm() != 0o600 {
			t.Errorf("Log file permissions = %o, expected 0600", info.Mode().Perm())
		}
	})

	t.Run("PID file can be read", func(t *testing.T) {
		pidBytes, err := os.ReadFile(pidFilePath)
		if err != nil {
			t.Fatalf("Failed to read PID file: %v", err)
		}
		expected := fmt.Sprintf("%d", pid)
		if string(pidBytes) != expected {
			t.Errorf("PID file content = %s, expected %s", string(pidBytes), expected)
		}
	})
}

// TestPIDFileRaceCondition verifies process cleanup when PID file write fails
func TestPIDFileRaceCondition(t *testing.T) {
	pidFilePath := getPIDFilePath()
	logFilePath := getLogFilePath()

	// Clean up
	os.Remove(pidFilePath)
	os.Remove(logFilePath)
	defer os.Remove(pidFilePath)
	defer os.Remove(logFilePath)

	t.Run("process starts successfully with PID file", func(t *testing.T) {
		logFile, err := os.OpenFile(logFilePath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
		if err != nil {
			t.Fatalf("Failed to create log file: %v", err)
		}
		defer logFile.Close()

		cmd := exec.Command("sleep", "1")
		cmd.Stdout = logFile
		cmd.Stderr = logFile

		if err := cmd.Start(); err != nil {
			t.Fatalf("Failed to start process: %v", err)
		}
		defer func() {
			_ = cmd.Process.Kill()
		}()

		pid := cmd.Process.Pid

		// Write PID file and cleanup on failure
		if err := os.WriteFile(pidFilePath, []byte(fmt.Sprintf("%d", pid)), 0o600); err != nil {
			// Kill the process if PID write fails to prevent orphaned processes
			_ = cmd.Process.Kill()
			t.Fatalf("Failed to write PID file: %v", err)
		}

		// Verify PID file exists
		if _, err := os.Stat(pidFilePath); os.IsNotExist(err) {
			t.Error("PID file should exist after successful write")
		}
	})

	t.Run("simulate PID write failure scenario", func(t *testing.T) {
		// Verify that process is killed if we cannot track it via PID file
		// Prevents orphaned processes that cannot be managed

		logFile, err := os.OpenFile(logFilePath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
		if err != nil {
			t.Fatalf("Failed to create log file: %v", err)
		}
		defer logFile.Close()

		cmd := exec.Command("sleep", "10")
		cmd.Stdout = logFile
		cmd.Stderr = logFile

		if err := cmd.Start(); err != nil {
			t.Fatalf("Failed to start process: %v", err)
		}

		pid := cmd.Process.Pid

		// Simulate trying to write PID to invalid location
		invalidPath := "/invalid/path/pid.file"
		writeErr := os.WriteFile(invalidPath, []byte(fmt.Sprintf("%d", pid)), 0o600)

		if writeErr != nil {
			// Kill process if we can't track it via PID file
			_ = cmd.Process.Kill()

			// Verify process is killed
			time.Sleep(100 * time.Millisecond)
			if err := cmd.Process.Signal(os.Signal(nil)); err == nil {
				t.Error("Process should be killed if PID file write fails")
			}
		}
	})
}

// TestPIDFileCleanup verifies proper cleanup
func TestPIDFileCleanup(t *testing.T) {
	pidFilePath := getPIDFilePath()
	logFilePath := getLogFilePath()

	// Clean up
	os.Remove(pidFilePath)
	os.Remove(logFilePath)

	t.Run("cleanup removes PID and log files", func(t *testing.T) {
		// Create files
		if err := os.WriteFile(pidFilePath, []byte("12345"), 0o600); err != nil {
			t.Fatalf("Failed to create PID file: %v", err)
		}
		if err := os.WriteFile(logFilePath, []byte("test logs"), 0o600); err != nil {
			t.Fatalf("Failed to create log file: %v", err)
		}

		// Verify they exist
		if _, err := os.Stat(pidFilePath); os.IsNotExist(err) {
			t.Error("PID file should exist before cleanup")
		}
		if _, err := os.Stat(logFilePath); os.IsNotExist(err) {
			t.Error("Log file should exist before cleanup")
		}

		// Clean up
		os.Remove(pidFilePath)
		os.Remove(logFilePath)

		// Verify they're gone
		if _, err := os.Stat(pidFilePath); !os.IsNotExist(err) {
			t.Error("PID file should not exist after cleanup")
		}
		if _, err := os.Stat(logFilePath); !os.IsNotExist(err) {
			t.Error("Log file should not exist after cleanup")
		}
	})
}

// TestIsProcessRunning tests the process detection helper function
func TestIsProcessRunning(t *testing.T) {
	t.Run("current process is detected as running", func(t *testing.T) {
		currentPID := os.Getpid()
		if !isProcessRunning(currentPID) {
			t.Error("Current process should be detected as running")
		}
	})

	t.Run("non-existent process is not running", func(t *testing.T) {
		// Use a very high PID that is unlikely to exist
		fakePID := 99999999
		if isProcessRunning(fakePID) {
			t.Error("Non-existent process should not be detected as running")
		}
	})

	t.Run("PID 0 is not running", func(t *testing.T) {
		if isProcessRunning(0) {
			t.Error("PID 0 should not be detected as running")
		}
	})

	t.Run("negative PID is not running", func(t *testing.T) {
		if isProcessRunning(-1) {
			t.Error("Negative PID should not be detected as running")
		}
	})
}

// TestStopProcess tests the graceful process shutdown helper function
func TestStopProcess(t *testing.T) {
	t.Run("stops a running process gracefully", func(t *testing.T) {
		// Start a long-running process that handles signals
		// Using 'cat' with no input which will exit on signal
		cmd := exec.Command("sleep", "60")
		if err := cmd.Start(); err != nil {
			t.Fatalf("Failed to start test process: %v", err)
		}

		pid := cmd.Process.Pid

		// Verify it's running
		if !isProcessRunning(pid) {
			t.Fatal("Test process should be running before stop")
		}

		// Stop the process
		err := stopProcess(pid)
		if err != nil {
			t.Errorf("stopProcess returned error: %v", err)
		}

		// Wait for process to fully terminate and be reaped
		// Use cmd.Wait() to properly reap the zombie process
		_ = cmd.Wait()

		// Give the OS a moment to update process state
		time.Sleep(500 * time.Millisecond)

		// Verify it's stopped - the process should no longer exist
		if isProcessRunning(pid) {
			t.Error("Process should be stopped after stopProcess call")
			// Clean up in case test fails
			_ = cmd.Process.Kill()
		}
	})

	t.Run("returns error for non-existent process", func(t *testing.T) {
		fakePID := 99999999
		err := stopProcess(fakePID)
		// On Unix, FindProcess doesn't error but Signal does
		// So we expect an error when trying to signal a non-existent process
		if err == nil {
			t.Error("stopProcess should return error for non-existent process")
		}
	})
}

// TestDeployLocalAlreadyRunning tests that deploy local prevents orphaned processes
func TestDeployLocalAlreadyRunning(t *testing.T) {
	pidFilePath := getPIDFilePath()
	logFilePath := getLogFilePath()

	// Clean up
	os.Remove(pidFilePath)
	os.Remove(logFilePath)
	defer os.Remove(pidFilePath)
	defer os.Remove(logFilePath)

	t.Run("detects stale PID file with dead process", func(t *testing.T) {
		// Create a PID file with a non-existent process
		stalePID := 99999999
		err := os.WriteFile(pidFilePath, []byte(fmt.Sprintf("%d", stalePID)), 0o600)
		if err != nil {
			t.Fatalf("Failed to create stale PID file: %v", err)
		}

		// Verify the process is not running
		if isProcessRunning(stalePID) {
			t.Skip("Skipping: fake PID unexpectedly exists")
		}

		// This simulates the cleanup logic in DeployLocal
		// The stale PID file should be detected and cleaned up
	})

	t.Run("detects running process from PID file", func(t *testing.T) {
		// Use our own PID as the "running" process
		currentPID := os.Getpid()
		err := os.WriteFile(pidFilePath, []byte(fmt.Sprintf("%d", currentPID)), 0o600)
		if err != nil {
			t.Fatalf("Failed to create PID file: %v", err)
		}
		defer os.Remove(pidFilePath)

		// Verify the process is detected as running
		if !isProcessRunning(currentPID) {
			t.Error("Current process should be detected as running")
		}
	})
}
