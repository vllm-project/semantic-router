package auth

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"unicode/utf8"

	"golang.org/x/text/unicode/norm"
)

const (
	maximumBlocklistFileBytes = 8 * 1024 * 1024
	maximumBlocklistLineBytes = 4096
	maximumBlocklistEntries   = 250_000

	PasswordSecurityProfileDevelopment = "development"
	PasswordSecurityProfileProduction  = "production"

	MinimumProductionPasswordBlocklistEntries = 10_000
)

// PasswordBlocklistLoadConfig selects the deployment security posture and,
// for an external corpus, the exact file bytes expected by the operator.
type PasswordBlocklistLoadConfig struct {
	Profile        string
	Path           string
	ExpectedSHA256 string
}

// PasswordBlocklistMetadata is safe to expose to authenticated operators. It
// proves which policy was loaded without exposing corpus values or its path.
type PasswordBlocklistMetadata struct {
	Profile    string `json:"profile"`
	EntryCount int    `json:"entryCount"`
	SHA256     string `json:"sha256"`
}

// LoadPasswordBlocklist preserves the development-profile compatibility API.
// Production callers must use LoadPasswordBlocklistForProfile so an external
// corpus and its expected digest cannot be omitted accidentally.
func LoadPasswordBlocklist(path string) (PasswordBlocklist, error) {
	blocklist, _, err := LoadPasswordBlocklistForProfile(PasswordBlocklistLoadConfig{
		Profile: PasswordSecurityProfileDevelopment,
		Path:    path,
	})
	return blocklist, err
}

// LoadPasswordBlocklistForProfile loads a bounded newline-delimited corpus.
// Production requires an external regular file, at least 10,000 unique NFC
// entries, and an exact SHA-256 digest of the original file bytes. Empty lines
// and lines whose first byte is # are ignored. Leading and trailing whitespace
// is otherwise significant. Loading fails closed; entries are never logged.
func LoadPasswordBlocklistForProfile(
	config PasswordBlocklistLoadConfig,
) (PasswordBlocklist, PasswordBlocklistMetadata, error) {
	profile := strings.TrimSpace(config.Profile)
	if profile == "" {
		profile = PasswordSecurityProfileDevelopment
	}
	if profile != PasswordSecurityProfileDevelopment && profile != PasswordSecurityProfileProduction {
		return nil, PasswordBlocklistMetadata{}, fmt.Errorf(
			"password security profile must be %q or %q",
			PasswordSecurityProfileDevelopment,
			PasswordSecurityProfileProduction,
		)
	}

	path := strings.TrimSpace(config.Path)
	expectedDigest, err := normalizeExpectedBlocklistSHA256(config.ExpectedSHA256)
	if err != nil {
		return nil, PasswordBlocklistMetadata{}, err
	}
	if path == "" {
		return loadBuiltInBlocklist(profile, expectedDigest)
	}
	if profile == PasswordSecurityProfileProduction && expectedDigest == "" {
		return nil, PasswordBlocklistMetadata{}, errors.New(
			"production password security profile requires a password blocklist SHA-256",
		)
	}

	entries, actualDigest, err := loadExternalBlocklist(path)
	if err != nil {
		return nil, PasswordBlocklistMetadata{}, err
	}
	if expectedDigest != "" && actualDigest != expectedDigest {
		return nil, PasswordBlocklistMetadata{}, errors.New(
			"password blocklist SHA-256 does not match the configured digest",
		)
	}
	if profile == PasswordSecurityProfileProduction &&
		len(entries) < MinimumProductionPasswordBlocklistEntries {
		return nil, PasswordBlocklistMetadata{}, fmt.Errorf(
			"production password blocklist has %d unique NFC entries; at least %d are required",
			len(entries),
			MinimumProductionPasswordBlocklistEntries,
		)
	}

	return newLocalPasswordBlocklist(entries...), PasswordBlocklistMetadata{
		Profile:    profile,
		EntryCount: len(entries),
		SHA256:     actualDigest,
	}, nil
}

func loadBuiltInBlocklist(
	profile string,
	expectedDigest string,
) (PasswordBlocklist, PasswordBlocklistMetadata, error) {
	if profile == PasswordSecurityProfileProduction {
		return nil, PasswordBlocklistMetadata{}, errors.New(
			"production password security profile requires an external password blocklist",
		)
	}
	if expectedDigest != "" {
		return nil, PasswordBlocklistMetadata{}, errors.New(
			"password blocklist SHA-256 requires an external password blocklist",
		)
	}
	blocklist := newLocalPasswordBlocklist()
	return blocklist, PasswordBlocklistMetadata{
		Profile:    profile,
		EntryCount: len(blocklist.exactValues),
	}, nil
}

func loadExternalBlocklist(path string) ([]string, string, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, "", fmt.Errorf("open password blocklist: %w", err)
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return nil, "", fmt.Errorf("stat password blocklist: %w", err)
	}
	if !info.Mode().IsRegular() {
		return nil, "", errors.New("password blocklist must be a regular file")
	}
	if info.Size() > maximumBlocklistFileBytes {
		return nil, "", fmt.Errorf(
			"password blocklist exceeds %d bytes",
			maximumBlocklistFileBytes,
		)
	}

	hasher := sha256.New()
	entries, err := scanPasswordBlocklist(io.TeeReader(file, hasher))
	if err != nil {
		return nil, "", err
	}
	if len(entries) == 0 {
		return nil, "", errors.New("password blocklist contains no usable entries")
	}
	return entries, hex.EncodeToString(hasher.Sum(nil)), nil
}

func normalizeExpectedBlocklistSHA256(value string) (string, error) {
	normalized := strings.ToLower(strings.TrimSpace(value))
	if normalized == "" {
		return "", nil
	}
	decoded, err := hex.DecodeString(normalized)
	if err != nil || len(decoded) != sha256.Size {
		return "", errors.New("password blocklist SHA-256 must be exactly 64 hexadecimal characters")
	}
	return normalized, nil
}

func scanPasswordBlocklist(reader io.Reader) ([]string, error) {
	limited := &io.LimitedReader{R: reader, N: maximumBlocklistFileBytes + 1}
	entries := make([]string, 0, 1024)
	seen := make(map[string]struct{}, 1024)
	usableEntries := 0
	scanner := bufio.NewScanner(limited)
	// The explicit byte check defines the line limit. The extra two scanner
	// bytes accommodate the delimiter (including CRLF) at the exact boundary.
	scanner.Buffer(make([]byte, 4096), maximumBlocklistLineBytes+2)
	for scanner.Scan() {
		if len(scanner.Bytes()) > maximumBlocklistLineBytes {
			return nil, fmt.Errorf(
				"password blocklist line exceeds %d bytes",
				maximumBlocklistLineBytes,
			)
		}
		entry := scanner.Text()
		if entry == "" || strings.HasPrefix(entry, "#") {
			continue
		}
		if !utf8.ValidString(entry) {
			return nil, errors.New("password blocklist contains invalid Unicode")
		}
		usableEntries++
		if usableEntries > maximumBlocklistEntries {
			return nil, fmt.Errorf("password blocklist exceeds %d entries", maximumBlocklistEntries)
		}
		normalized := norm.NFC.String(entry)
		if _, found := seen[normalized]; found {
			continue
		}
		seen[normalized] = struct{}{}
		entries = append(entries, normalized)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("read password blocklist: %w", err)
	}
	if limited.N == 0 {
		return nil, fmt.Errorf("password blocklist exceeds %d bytes", maximumBlocklistFileBytes)
	}
	return entries, nil
}
