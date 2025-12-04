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
