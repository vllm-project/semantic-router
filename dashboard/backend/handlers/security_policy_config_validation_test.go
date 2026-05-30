package handlers

import (
	"bytes"
	"os"
	"testing"
)

func TestValidateMergedSecurityConfigAcceptsPolicyMerge(t *testing.T) {
	dir := t.TempDir()
	configPath := createValidTestConfig(t, dir)
	base, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("read config: %v", err)
	}

	fragment := GenerateRouterFragment(&SecurityPolicyConfig{
		RoleMappings: []RoleMapping{
			{
				Name:      "premium",
				Subjects:  []Subject{{Kind: "Group", Name: "paying-users"}},
				Role:      "premium_tier",
				ModelRefs: []string{"test-model"},
				Priority:  10,
			},
		},
	})
	fragmentYAML, err := toCanonicalYAML(fragment)
	if err != nil {
		t.Fatalf("marshal fragment: %v", err)
	}
	merged, err := mergeDeployPayload(base, DeployRequest{YAML: string(fragmentYAML)})
	if err != nil {
		t.Fatalf("merge fragment: %v", err)
	}

	if err := validateMergedSecurityConfig(merged); err != nil {
		t.Fatalf("expected merged config to validate: %v", err)
	}
}

func TestApplySecurityFragmentRejectsInvalidMergedConfigBeforeWrite(t *testing.T) {
	dir := t.TempDir()
	configPath := createValidTestConfig(t, dir)
	base, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("read config: %v", err)
	}
	invalidBase := bytes.Replace(
		base,
		[]byte("endpoint: 127.0.0.1:8000"),
		[]byte("endpoint: http://127.0.0.1:8000"),
		1,
	)
	if bytes.Equal(invalidBase, base) {
		t.Fatal("test fixture did not replace backend endpoint")
	}
	if writeErr := os.WriteFile(configPath, invalidBase, 0o644); writeErr != nil {
		t.Fatalf("write invalid config: %v", writeErr)
	}

	previousPath, previousDir := securityPolicyConfigPath, securityPolicyConfigDir
	SetSecurityPolicyConfigPaths(configPath, dir)
	t.Cleanup(func() {
		SetSecurityPolicyConfigPaths(previousPath, previousDir)
	})

	fragment := GenerateRouterFragment(&SecurityPolicyConfig{
		RoleMappings: []RoleMapping{
			{
				Name:      "premium",
				Subjects:  []Subject{{Kind: "Group", Name: "paying-users"}},
				Role:      "premium_tier",
				ModelRefs: []string{"test-model"},
				Priority:  10,
			},
		},
	})

	if applySecurityFragment(fragment) {
		t.Fatal("expected invalid merged config to be rejected")
	}

	after, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("read config after apply: %v", err)
	}
	if !bytes.Equal(after, invalidBase) {
		t.Fatal("invalid merged config should not be written")
	}
}
