package extproc

import (
	"path/filepath"
	"testing"
)

func TestShouldReloadForConfigEvent(t *testing.T) {
	cfgDir := "/app/config"
	cfgFile := filepath.Join(cfgDir, "config.yaml")

	tests := []struct {
		name      string
		eventPath string
		want      bool
	}{
		{
			name:      "exact config file path",
			eventPath: cfgFile,
			want:      true,
		},
		{
			name:      "config basename in watched dir",
			eventPath: filepath.Join(cfgDir, "config.yaml"),
			want:      true,
		},
		{
			name:      "configmap data symlink swap",
			eventPath: filepath.Join(cfgDir, "..data_tmp"),
			want:      true,
		},
		{
			name:      "startup status writability probe",
			eventPath: filepath.Join(cfgDir, ".vllm-sr-write-check-123"),
			want:      false,
		},
		{
			name:      "unrelated sibling file",
			eventPath: filepath.Join(cfgDir, "router-runtime.json"),
			want:      false,
		},
		{
			name:      "event in other directory",
			eventPath: "/tmp/config.yaml",
			want:      false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := shouldReloadForConfigEvent(cfgFile, cfgDir, tc.eventPath)
			if got != tc.want {
				t.Fatalf("shouldReloadForConfigEvent(%q) = %t, want %t", tc.eventPath, got, tc.want)
			}
		})
	}
}
