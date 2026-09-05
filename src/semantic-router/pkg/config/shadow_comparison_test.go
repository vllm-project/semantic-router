package config

import (
	"testing"
	"time"
)

func TestShadowComparisonIsEnabled(t *testing.T) {
	if got := (ShadowComparisonConfig{}).IsEnabled(); got {
		t.Fatal("disabled default must not be enabled")
	}
	onlyEnabled := ShadowComparisonConfig{Enabled: true}
	if got := onlyEnabled.IsEnabled(); got {
		t.Fatal("enabled with no arms must not be enabled")
	}
	halfArmed := ShadowComparisonConfig{
		Enabled: true,
		Arms:    []ShadowArmConfig{{Name: "a", Model: "m"}}, // missing endpoint
	}
	if got := halfArmed.IsEnabled(); got {
		t.Fatal("arm missing endpoint must disable the feature")
	}
	ok := ShadowComparisonConfig{
		Enabled: true,
		Arms:    []ShadowArmConfig{{Name: "a", Model: "m", Endpoint: "http://x"}},
	}
	if got := ok.IsEnabled(); !got {
		t.Fatal("fully armed config must be enabled")
	}
}

func TestShadowComparisonGetMaxWait(t *testing.T) {
	if got := (ShadowComparisonConfig{}).GetMaxWait(); got != DefaultShadowMaxWait {
		t.Fatalf("default wait = %v, want %v", got, DefaultShadowMaxWait)
	}
	if got := (ShadowComparisonConfig{MaxWaitMS: 250}).GetMaxWait(); got != 250*time.Millisecond {
		t.Fatalf("configured wait = %v", got)
	}
}
