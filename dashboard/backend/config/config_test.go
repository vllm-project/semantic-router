package config

import (
	"path/filepath"
	"slices"
	"testing"
)

func TestResolveConfigPathsHonorsDashboardBaseDirectoryOverride(t *testing.T) {
	configRoot := t.TempDir()
	baseDirectory := t.TempDir()
	t.Setenv("DASHBOARD_CONFIG_DIR", baseDirectory)
	cfg := &Config{ConfigFile: filepath.Join(configRoot, ".vllm-sr", "runtime-config.stack.yaml")}
	if err := resolveConfigPaths(cfg); err != nil {
		t.Fatal(err)
	}
	want, err := filepath.Abs(baseDirectory)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.ConfigDir != want {
		t.Fatalf("ConfigDir = %q, want %q", cfg.ConfigDir, want)
	}
	if cfg.AbsConfigPath == "" || filepath.Base(cfg.AbsConfigPath) != "runtime-config.stack.yaml" {
		t.Fatalf("AbsConfigPath = %q", cfg.AbsConfigPath)
	}
}

func TestParseAllowedOrigins(t *testing.T) {
	cases := []struct {
		name string
		raw  string
		want []string
	}{
		{name: "empty"},
		{name: "only separators", raw: " , , "},
		{name: "single", raw: "http://localhost:3001", want: []string{"http://localhost:3001"}},
		{
			name: "trims, lowercases, drops blanks",
			raw:  " HTTP://Dash.Example ,, https://Other.Example:8443 ",
			want: []string{"http://dash.example", "https://other.example:8443"},
		},
		{name: "drops a trailing slash", raw: "http://localhost:3001/", want: []string{"http://localhost:3001"}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := parseAllowedOrigins(tc.raw); !slices.Equal(got, tc.want) {
				t.Fatalf("parseAllowedOrigins(%q) = %v, want %v", tc.raw, got, tc.want)
			}
		})
	}
}
