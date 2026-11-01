package config

import (
	"strings"
	"testing"
	"time"
)

func TestBackendTelemetrySettingsDefaults(t *testing.T) {
	settings, err := BackendTelemetryConfig{Enabled: true}.Settings()
	if err != nil {
		t.Fatal(err)
	}
	want := BackendTelemetrySettings{PollInterval: 2 * time.Second, RequestTimeout: time.Second, TTL: 6 * time.Second}
	if settings != want {
		t.Fatalf("Settings() = %+v, want %+v", settings, want)
	}
}

func TestBackendTelemetrySettingsOverrides(t *testing.T) {
	settings, err := BackendTelemetryConfig{PollInterval: "500ms", RequestTimeout: "250ms", TTL: "2s"}.Settings()
	if err != nil {
		t.Fatal(err)
	}
	want := BackendTelemetrySettings{PollInterval: 500 * time.Millisecond, RequestTimeout: 250 * time.Millisecond, TTL: 2 * time.Second}
	if settings != want {
		t.Fatalf("Settings() = %+v, want %+v", settings, want)
	}
}

func TestBackendTelemetrySettingsRejectInvalidDurations(t *testing.T) {
	for _, tc := range []struct {
		cfg  BackendTelemetryConfig
		want string
	}{
		{BackendTelemetryConfig{PollInterval: "soon"}, "poll_interval"},
		{BackendTelemetryConfig{RequestTimeout: "0s"}, "request_timeout"},
		{BackendTelemetryConfig{TTL: "-1s"}, "ttl must be a positive"},
		{BackendTelemetryConfig{PollInterval: "5s", TTL: "1s"}, "ttl must not be shorter"},
	} {
		if _, err := tc.cfg.Settings(); err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Fatalf("Settings(%+v) = %v, want %q", tc.cfg, err, tc.want)
		}
	}
}
