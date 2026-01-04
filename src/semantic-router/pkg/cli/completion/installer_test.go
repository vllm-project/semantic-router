package completion

import (
	"runtime"
	"strings"
	"testing"
)

func TestGetPowerShellPaths_Platform(t *testing.T) {
	paths := getPowerShellPaths()

	if len(paths) != 2 {
		t.Errorf("expected 2 paths, got %d", len(paths))
	}

	if runtime.GOOS == "windows" {
		// Should contain Windows paths
		for _, p := range paths {
			if !strings.Contains(p, "\\") {
				t.Errorf("Windows path should use backslashes: %s", p)
			}
		}

		foundDocs := false
		foundPF := false
		for _, p := range paths {
			if strings.Contains(p, "Documents") && strings.Contains(p, "PowerShell") {
				foundDocs = true
			}
			if strings.Contains(p, "Program Files") || strings.Contains(p, "PowerShell") {
				foundPF = true
			}
		}
		if !foundDocs || !foundPF {
			t.Errorf("Windows paths missing expected components: %v", paths)
		}
	} else {
		// Should NOT contain Windows paths on Linux/macOS
		for _, p := range paths {
			if strings.Contains(p, "C:") || strings.Contains(p, "Documents") || strings.Contains(p, "\\") {
				t.Errorf("Non-Windows path should not contain Windows-style paths: %s", p)
			}
		}
		// Should use .config directory
		if !strings.Contains(paths[0], ".config") {
			t.Errorf("Linux/macOS user path should contain .config: %s", paths[0])
		}
	}
}

func TestGetPowerShellPaths_ContainsVsr(t *testing.T) {
	paths := getPowerShellPaths()

	for _, p := range paths {
		if !strings.Contains(p, "vsr") {
			t.Errorf("PowerShell path should contain 'vsr': %s", p)
		}
	}
}
