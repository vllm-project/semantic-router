package config

import (
	"strings"
	"testing"
)

func TestValidateCanonicalVersion(t *testing.T) {
	tests := []struct {
		name    string
		version string
		wantErr bool
	}{
		{"supported", SupportedCanonicalVersion, false},
		{"absent is accepted for compatibility", "", false},
		{"whitespace only is treated as absent", "   ", false},
		{"older contract", "v0.1", true},
		{"future contract", "v99", true},
		{"unprefixed", "0.3", true},
		{"not a version", "latest", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateCanonicalVersion(tt.version)
			if (err != nil) != tt.wantErr {
				t.Fatalf("validateCanonicalVersion(%q) error = %v, wantErr %v", tt.version, err, tt.wantErr)
			}
			if err != nil && !strings.HasPrefix(err.Error(), "version: ") {
				t.Errorf("error lacks a version field path: %v", err)
			}
		})
	}
}

// TestCanonicalContractRejectsUnsupportedVersion covers the boundary rather than the
// helper: an unsupported version must fail before the document is interpreted.
func TestCanonicalContractRejectsUnsupportedVersion(t *testing.T) {
	canonical := &CanonicalConfig{Version: "v0.1"}

	if err := validateCanonicalContract(canonical); err == nil {
		t.Fatal("validateCanonicalContract accepted version v0.1")
	}

	canonical.Version = SupportedCanonicalVersion
	if err := validateCanonicalContract(canonical); err != nil {
		t.Fatalf("validateCanonicalContract rejected the supported version: %v", err)
	}
}

// TestExportStampsSupportedVersion keeps the exporter and the gate on one constant,
// so a document this build writes is one it will read back.
func TestExportStampsSupportedVersion(t *testing.T) {
	for _, cfg := range []*RouterConfig{nil, {}} {
		got := CanonicalConfigFromRouterConfig(cfg).Version
		if got != SupportedCanonicalVersion {
			t.Errorf("exported version = %q, want %q", got, SupportedCanonicalVersion)
		}
		if err := validateCanonicalVersion(got); err != nil {
			t.Errorf("exported version does not pass the gate: %v", err)
		}
	}
}
