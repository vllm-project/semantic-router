package deployment

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

// TestCrossPlatformPaths verifies cross-platform path handling using os.TempDir
func TestCrossPlatformPaths(t *testing.T) {
	t.Run("getPIDFilePath returns absolute path", func(t *testing.T) {
		pidFilePath := getPIDFilePath()
		if pidFilePath == "" {
			t.Error("getPIDFilePath() returned empty string")
		}
		if !filepath.IsAbs(pidFilePath) {
			t.Errorf("getPIDFilePath() should return absolute path, got: %s", pidFilePath)
		}
	})

	t.Run("getLogFilePath returns absolute path", func(t *testing.T) {
		logFilePath := getLogFilePath()
		if logFilePath == "" {
			t.Error("getLogFilePath() returned empty string")
		}
		if !filepath.IsAbs(logFilePath) {
			t.Errorf("getLogFilePath() should return absolute path, got: %s", logFilePath)
		}
	})

	t.Run("paths are user-specific", func(t *testing.T) {
		pidFilePath := getPIDFilePath()
		expectedSubstring := fmt.Sprintf("-%d.", os.Getuid())
		// Use Go's strings package to check for substring
		found := false
		for i := 0; i <= len(pidFilePath)-len(expectedSubstring); i++ {
			if i+len(expectedSubstring) <= len(pidFilePath) && pidFilePath[i:i+len(expectedSubstring)] == expectedSubstring {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("pidFilePath should contain UID (%d), got: %s", os.Getuid(), pidFilePath)
		}
	})

	t.Run("paths use temp directory", func(t *testing.T) {
		pidFilePath := getPIDFilePath()
		tempDir := os.TempDir()
		// Check if path starts with temp directory
		if len(pidFilePath) < len(tempDir) {
			t.Errorf("pidFilePath too short, got: %s", pidFilePath)
			return
		}
		if pidFilePath[:len(tempDir)] != tempDir {
			t.Errorf("pidFilePath should start with temp directory (%s), got: %s", tempDir, pidFilePath)
		}
	})
}

// TestPathFunctions verifies path helper functions work correctly
func TestPathFunctions(t *testing.T) {
	t.Run("paths are consistent", func(t *testing.T) {
		pid1 := getPIDFilePath()
		pid2 := getPIDFilePath()
		if pid1 != pid2 {
			t.Errorf("getPIDFilePath() should return consistent results, got %s and %s", pid1, pid2)
		}
	})

	t.Run("PID and log paths are different", func(t *testing.T) {
		pidPath := getPIDFilePath()
		logPath := getLogFilePath()
		if pidPath == logPath {
			t.Error("PID and log file paths should be different")
		}
	})
}
