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
		{"supported", CanonicalConfigVersion, false},
		{"absent is accepted for compatibility", "", false},
		{"whitespace only is not a contract", "   ", true},
		{"padded supported version", " " + CanonicalConfigVersion + " ", true},
		{"tab padded", "\t" + CanonicalConfigVersion, true},
		{"older contract", "v0.1", true},
		{"future contract", "v99", true},
		{"unprefixed", "0.3", true},
		{"not a version", "latest", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateCanonicalVersion(&CanonicalConfig{Version: tt.version})
			if (err != nil) != tt.wantErr {
				t.Fatalf("validateCanonicalVersion(%q) error = %v, wantErr %v", tt.version, err, tt.wantErr)
			}
			if err != nil && !strings.HasPrefix(err.Error(), "version: ") {
				t.Errorf("error lacks a version field path: %v", err)
			}
		})
	}
}

func TestValidateCanonicalVersionRejectsNil(t *testing.T) {
	if err := validateCanonicalVersion(nil); err == nil {
		t.Fatal("validateCanonicalVersion(nil) = nil, want an error")
	}
}

// A release that bumps the written contract lists the outgoing one in
// acceptedCanonicalVersions, so configs written for it keep loading.
func TestValidateCanonicalVersionAcceptsRetainedContracts(t *testing.T) {
	restore := acceptedCanonicalVersions
	t.Cleanup(func() { acceptedCanonicalVersions = restore })
	acceptedCanonicalVersions = []string{"v0.4", "v0.3"}

	for _, v := range acceptedCanonicalVersions {
		if err := validateCanonicalVersion(&CanonicalConfig{Version: v}); err != nil {
			t.Errorf("validateCanonicalVersion(%q) = %v, want accepted", v, err)
		}
	}

	err := validateCanonicalVersion(&CanonicalConfig{Version: "v0.2"})
	if err == nil {
		t.Fatal(`validateCanonicalVersion("v0.2") accepted a dropped contract`)
	}
	if !strings.Contains(err.Error(), "v0.4, v0.3") {
		t.Errorf("error does not name what this build reads: %v", err)
	}
}

// TestCanonicalContractRejectsUnsupportedVersion covers the boundary rather than the
// helper: an unsupported version must fail before the document is interpreted.
func TestCanonicalContractRejectsUnsupportedVersion(t *testing.T) {
	canonical := &CanonicalConfig{Version: "v0.1"}

	if err := validateCanonicalContract(canonical); err == nil {
		t.Fatal("validateCanonicalContract accepted version v0.1")
	}

	canonical.Version = CanonicalConfigVersion
	if err := validateCanonicalContract(canonical); err != nil {
		t.Fatalf("validateCanonicalContract rejected the supported version: %v", err)
	}
}

// TestExportStampsSupportedVersion keeps the exporter and the gate on one constant,
// so a document this build writes is one it will read back.
func TestExportStampsSupportedVersion(t *testing.T) {
	for _, cfg := range []*RouterConfig{nil, {}} {
		exported := CanonicalConfigFromRouterConfig(cfg)
		if exported.Version != CanonicalConfigVersion {
			t.Errorf("exported version = %q, want %q", exported.Version, CanonicalConfigVersion)
		}
		if err := validateCanonicalVersion(&exported); err != nil {
			t.Errorf("exported version does not pass the gate: %v", err)
		}
	}
}

// The version is read straight off the parsed YAML map, so a document written
// for another contract is refused before any normalizer rewrites it.
func TestValidateRawCanonicalVersion(t *testing.T) {
	tests := []struct {
		name    string
		raw     map[string]interface{}
		wantErr bool
	}{
		{"supported", map[string]interface{}{"version": CanonicalConfigVersion}, false},
		{"key absent", map[string]interface{}{}, false},
		{"empty string", map[string]interface{}{"version": ""}, false},
		{"whitespace only", map[string]interface{}{"version": "   "}, true},
		{"padded", map[string]interface{}{"version": " " + CanonicalConfigVersion + " "}, true},
		{"older contract", map[string]interface{}{"version": "v0.2"}, true},
		{"not a string", map[string]interface{}{"version": 0.3}, true},
		{"null", map[string]interface{}{"version": nil}, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateRawCanonicalVersion(tt.raw)
			if (err != nil) != tt.wantErr {
				t.Fatalf("ValidateRawCanonicalVersion(%v) error = %v, wantErr %v", tt.raw, err, tt.wantErr)
			}
			if err != nil && !strings.HasPrefix(err.Error(), "version: ") {
				t.Errorf("error lacks a version field path: %v", err)
			}
		})
	}
}

// An unsupported version wins over every other complaint the document attracts.
// Otherwise a file written for another contract is reported as a bad field in
// this one, which sends the reader to the wrong fix.
func TestUnsupportedVersionOutranksOtherRejections(t *testing.T) {
	// routing.models is a deprecated field, so this document fails twice over.
	document := []byte("version: v0.2\nrouting:\n  models:\n    - name: legacy\n")

	_, err := ParseYAMLBytes(document)
	if err == nil {
		t.Fatal("ParseYAMLBytes accepted an unsupported version")
	}
	if !strings.HasPrefix(err.Error(), "version: ") {
		t.Fatalf("a deprecated field was reported ahead of the version gate: %v", err)
	}

	// The same document on the supported contract still fails, on the field.
	supported := []byte("version: " + CanonicalConfigVersion + "\nrouting:\n  models:\n    - name: legacy\n")
	_, err = ParseYAMLBytes(supported)
	if err == nil {
		t.Fatal("ParseYAMLBytes accepted a deprecated field")
	}
	if strings.HasPrefix(err.Error(), "version: ") {
		t.Fatalf("supported version was reported as a version failure: %v", err)
	}
}

// The generated JSON Schema gates on the same list the Router reads, so the CLI
// cannot reject a document the Router loads.
func TestAcceptedCanonicalVersionsIsACopy(t *testing.T) {
	accepted := AcceptedCanonicalVersions()
	if len(accepted) == 0 || accepted[0] != CanonicalConfigVersion {
		t.Fatalf("AcceptedCanonicalVersions() = %v, want the written contract first", accepted)
	}
	accepted[0] = "tampered"
	if acceptedCanonicalVersions[0] == "tampered" {
		t.Fatal("AcceptedCanonicalVersions() exposed the package slice")
	}
}
