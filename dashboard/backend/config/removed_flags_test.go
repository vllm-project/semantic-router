package config

import (
	"flag"
	"os"
	"slices"
	"testing"
)

func loadConfigWithArgs(t *testing.T, args ...string) *Config {
	t.Helper()
	oldFlags, oldArgs := flag.CommandLine, os.Args
	t.Cleanup(func() { flag.CommandLine, os.Args = oldFlags, oldArgs })
	flag.CommandLine = flag.NewFlagSet("removed-flags-test", flag.ContinueOnError)
	os.Args = append([]string{"dashboard"}, args...)
	cfg, err := LoadConfig()
	if err != nil {
		t.Fatalf("LoadConfig(%v): %v", args, err)
	}
	return cfg
}

func clearOpenClawEnvironment(t *testing.T) {
	t.Helper()
	for _, setting := range removedOpenClawSettings {
		t.Setenv(setting.env, "")
	}
}

func TestRemovedOpenClawFlagsStillParseAndAreReported(t *testing.T) {
	clearOpenClawEnvironment(t)
	t.Setenv("OPENCLAW_ENABLED", "false")
	cfg := loadConfigWithArgs(t,
		"-openclaw", "-port", "8701",
		"-openclaw-url=http://localhost:18788",
		"--openclaw-data", "./data/openclaw",
		"-openclaw-token=unused",
	)
	if cfg.Port != "8701" {
		t.Fatalf("Port = %q, want 8701: the boolean -openclaw flag must not take the next argument", cfg.Port)
	}
	want := []string{"-openclaw", "OPENCLAW_ENABLED", "-openclaw-url", "-openclaw-data", "-openclaw-token"}
	if !slices.Equal(cfg.IgnoredOpenClawSettings, want) {
		t.Fatalf("IgnoredOpenClawSettings = %v, want %v", cfg.IgnoredOpenClawSettings, want)
	}
}

func TestNoOpenClawSettingsReportsNothing(t *testing.T) {
	clearOpenClawEnvironment(t)
	cfg := loadConfigWithArgs(t, "-port", "8702")
	if len(cfg.IgnoredOpenClawSettings) != 0 {
		t.Fatalf("IgnoredOpenClawSettings = %v, want none", cfg.IgnoredOpenClawSettings)
	}
}
