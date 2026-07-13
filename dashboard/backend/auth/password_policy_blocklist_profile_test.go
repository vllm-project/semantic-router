package auth

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const productionBlocklistCanary = "known breached canary café credential"

func TestProductionPasswordBlocklistRequiresVerifiedNFCUniqueCorpusAndBlocksCanary(
	t *testing.T,
) {
	t.Parallel()

	decomposedCanary := strings.ReplaceAll(productionBlocklistCanary, "é", "e\u0301")
	path, digest := writeProductionPasswordBlocklist(t, decomposedCanary)
	blocklist, metadata, err := LoadPasswordBlocklistForProfile(PasswordBlocklistLoadConfig{
		Profile:        PasswordSecurityProfileProduction,
		Path:           path,
		ExpectedSHA256: strings.ToUpper(digest),
	})
	if err != nil {
		t.Fatalf("LoadPasswordBlocklistForProfile() error = %v", err)
	}
	if metadata.Profile != PasswordSecurityProfileProduction ||
		metadata.EntryCount != MinimumProductionPasswordBlocklistEntries ||
		metadata.SHA256 != digest {
		t.Fatalf("metadata = %#v", metadata)
	}

	_, err = NewPasswordPolicy(blocklist).Validate(
		"person@example.com",
		productionBlocklistCanary,
	)
	var policyErr *PasswordPolicyError
	if !errors.As(err, &policyErr) || policyErr.Code != PasswordPolicyBlocked {
		t.Fatalf("canary Validate() error = %#v, want blocked policy error", err)
	}
}

func TestProductionPasswordBlocklistFailsClosed(t *testing.T) {
	t.Parallel()

	smallPath := filepath.Join(t.TempDir(), "small-passwords.txt")
	smallContents := []byte("one compromised credential\n")
	if err := os.WriteFile(smallPath, smallContents, 0o600); err != nil {
		t.Fatalf("write small blocklist: %v", err)
	}
	smallSum := sha256.Sum256(smallContents)
	smallDigest := hex.EncodeToString(smallSum[:])

	tests := []struct {
		name   string
		config PasswordBlocklistLoadConfig
		match  string
	}{
		{
			name:   "missing file",
			config: PasswordBlocklistLoadConfig{Profile: PasswordSecurityProfileProduction},
			match:  "requires an external",
		},
		{
			name: "missing digest",
			config: PasswordBlocklistLoadConfig{
				Profile: PasswordSecurityProfileProduction,
				Path:    smallPath,
			},
			match: "requires a password blocklist SHA-256",
		},
		{
			name: "malformed digest",
			config: PasswordBlocklistLoadConfig{
				Profile:        PasswordSecurityProfileProduction,
				Path:           smallPath,
				ExpectedSHA256: "not-a-digest",
			},
			match: "64 hexadecimal",
		},
		{
			name: "digest mismatch",
			config: PasswordBlocklistLoadConfig{
				Profile:        PasswordSecurityProfileProduction,
				Path:           smallPath,
				ExpectedSHA256: strings.Repeat("0", sha256.Size*2),
			},
			match: "does not match",
		},
		{
			name: "too few NFC unique entries",
			config: PasswordBlocklistLoadConfig{
				Profile:        PasswordSecurityProfileProduction,
				Path:           smallPath,
				ExpectedSHA256: smallDigest,
			},
			match: fmt.Sprintf("at least %d", MinimumProductionPasswordBlocklistEntries),
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, _, err := LoadPasswordBlocklistForProfile(test.config)
			if err == nil || !strings.Contains(err.Error(), test.match) {
				t.Fatalf("LoadPasswordBlocklistForProfile() error = %v, want %q", err, test.match)
			}
		})
	}
}

func writeProductionPasswordBlocklist(t *testing.T, additionalEntries ...string) (string, string) {
	t.Helper()

	var contents strings.Builder
	contents.WriteString(productionBlocklistCanary)
	contents.WriteByte('\n')
	for index := 1; index < MinimumProductionPasswordBlocklistEntries; index++ {
		contents.WriteString(fmt.Sprintf("offline compromised corpus entry %05d\n", index))
	}
	for _, entry := range additionalEntries {
		contents.WriteString(entry)
		contents.WriteByte('\n')
	}

	raw := []byte(contents.String())
	path := filepath.Join(t.TempDir(), "production-passwords.txt")
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		t.Fatalf("write production blocklist: %v", err)
	}
	sum := sha256.Sum256(raw)
	return path, hex.EncodeToString(sum[:])
}
