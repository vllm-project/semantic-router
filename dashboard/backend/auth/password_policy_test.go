package auth

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"golang.org/x/crypto/bcrypt"
)

func TestPasswordPolicyLengthNormalizationAndNoCompositionRule(t *testing.T) {
	t.Parallel()

	policy := NewPasswordPolicy(nil)
	tests := []struct {
		name     string
		password string
		wantCode string
	}{
		{name: "fourteen code points", password: strings.Repeat("a", 14), wantCode: PasswordPolicyTooShort},
		{name: "fifteen one-class code points", password: strings.Repeat("a", 15)},
		{name: "sixty-four multibyte code points", password: strings.Repeat("界", 64)},
		{name: "maximum accepted", password: strings.Repeat("z", maximumPasswordCodePoints)},
		{name: "over bounded maximum", password: strings.Repeat("z", maximumPasswordCodePoints+1), wantCode: PasswordPolicyTooLong},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			_, err := policy.Validate("person@example.com", test.password)
			if test.wantCode == "" {
				if err != nil {
					t.Fatalf("Validate() error = %v", err)
				}
				return
			}
			var policyErr *PasswordPolicyError
			if !errors.As(err, &policyErr) || policyErr.Code != test.wantCode {
				t.Fatalf("Validate() error = %#v, want policy code %q", err, test.wantCode)
			}
		})
	}

	decomposed := strings.Repeat("e\u0301", minimumPasswordCodePoints)
	composed := strings.Repeat("é", minimumPasswordCodePoints)
	normalized, err := policy.Validate("person@example.com", decomposed)
	if err != nil {
		t.Fatalf("Validate(decomposed) error = %v", err)
	}
	if normalized != composed {
		t.Fatalf("normalized password = %q, want NFC %q", normalized, composed)
	}
}

func TestPasswordPolicyBlocksCommonServiceAndAccountValues(t *testing.T) {
	t.Parallel()

	policy := NewPasswordPolicy(nil)
	tests := []struct {
		name     string
		email    string
		password string
	}{
		{name: "common separator variant", email: "person@example.com", password: "correct horse battery staple"},
		{name: "service specific", email: "person@example.com", password: "vLLM-Semantic-Router"},
		{name: "full account", email: "long-admin@example.com", password: "long-admin@example.com"},
		{name: "account local part", email: "long-administrator@example.com", password: "long_administrator"},
		{name: "complete whitespace value", email: "person@example.com", password: strings.Repeat(" ", minimumPasswordCodePoints)},
		{name: "complete repeated punctuation value", email: "person@example.com", password: strings.Repeat("!", minimumPasswordCodePoints)},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			_, err := policy.Validate(test.email, test.password)
			var policyErr *PasswordPolicyError
			if !errors.As(err, &policyErr) || policyErr.Code != PasswordPolicyBlocked {
				t.Fatalf("Validate() error = %#v, want blocked policy error", err)
			}
		})
	}
}

type testPasswordBlocklist struct{}

func (testPasswordBlocklist) Blocked(password, _ string) bool {
	return password == "custom blocklist password"
}

func TestPasswordPolicySupportsInjectedBlocklist(t *testing.T) {
	t.Parallel()

	_, err := NewPasswordPolicy(testPasswordBlocklist{}).Validate(
		"person@example.com",
		"custom blocklist password",
	)
	var policyErr *PasswordPolicyError
	if !errors.As(err, &policyErr) || policyErr.Code != PasswordPolicyBlocked {
		t.Fatalf("Validate() error = %#v, want injected blocklist rejection", err)
	}
}

func TestLoadPasswordBlocklistMergesBoundedLocalEntries(t *testing.T) {
	t.Parallel()

	path := filepath.Join(t.TempDir(), "passwords.txt")
	if err := os.WriteFile(
		path,
		[]byte("# managed offline breach corpus\norganization compromised credential\n"),
		0o600,
	); err != nil {
		t.Fatalf("write blocklist: %v", err)
	}
	blocklist, err := LoadPasswordBlocklist(path)
	if err != nil {
		t.Fatalf("LoadPasswordBlocklist() error = %v", err)
	}
	policy := NewPasswordPolicy(blocklist)
	for _, password := range []string{
		"organization compromised credential",
		"password-password",
	} {
		_, err := policy.Validate("person@example.com", password)
		var policyErr *PasswordPolicyError
		if !errors.As(err, &policyErr) || policyErr.Code != PasswordPolicyBlocked {
			t.Fatalf("Validate(%q) error = %#v, want blocked", password, err)
		}
	}
}

func TestLoadPasswordBlocklistPreservesWhitespaceBearingCompleteValues(t *testing.T) {
	t.Parallel()

	allSpaces := strings.Repeat(" ", minimumPasswordCodePoints+1)
	spaceBearing := "  organization exact credential  "
	path := filepath.Join(t.TempDir(), "passwords.txt")
	contents := "# exact offline corpus\n" + allSpaces + "\n" + spaceBearing + "\n"
	if err := os.WriteFile(path, []byte(contents), 0o600); err != nil {
		t.Fatalf("write blocklist: %v", err)
	}
	blocklist, err := LoadPasswordBlocklist(path)
	if err != nil {
		t.Fatalf("LoadPasswordBlocklist() error = %v", err)
	}
	policy := NewPasswordPolicy(blocklist)
	for _, password := range []string{allSpaces, spaceBearing} {
		_, err := policy.Validate("person@example.com", password)
		var policyErr *PasswordPolicyError
		if !errors.As(err, &policyErr) || policyErr.Code != PasswordPolicyBlocked {
			t.Fatalf("Validate(%q) error = %#v, want exact-value rejection", password, err)
		}
	}
}

func TestLoadPasswordBlocklistRejectsUnsafeInputs(t *testing.T) {
	t.Parallel()

	t.Run("missing", func(t *testing.T) {
		if _, err := LoadPasswordBlocklist(filepath.Join(t.TempDir(), "missing")); err == nil {
			t.Fatal("LoadPasswordBlocklist() accepted missing file")
		}
	})
	t.Run("directory", func(t *testing.T) {
		if _, err := LoadPasswordBlocklist(t.TempDir()); err == nil {
			t.Fatal("LoadPasswordBlocklist() accepted directory")
		}
	})
	for _, test := range []struct {
		name     string
		contents string
	}{
		{name: "empty corpus", contents: ""},
		{name: "empty lines only", contents: "\n\n"},
		{name: "comments only", contents: "# generated corpus\n# no entries\n"},
	} {
		t.Run(test.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "passwords.txt")
			if err := os.WriteFile(path, []byte(test.contents), 0o600); err != nil {
				t.Fatalf("write blocklist: %v", err)
			}
			if _, err := LoadPasswordBlocklist(path); err == nil {
				t.Fatal("LoadPasswordBlocklist() accepted a configured corpus without entries")
			}
		})
	}
	t.Run("oversized file", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "oversized.txt")
		file, err := os.Create(path)
		if err != nil {
			t.Fatalf("create: %v", err)
		}
		if err := file.Truncate(maximumBlocklistFileBytes + 1); err != nil {
			_ = file.Close()
			t.Fatalf("truncate: %v", err)
		}
		if err := file.Close(); err != nil {
			t.Fatalf("close: %v", err)
		}
		if _, err := LoadPasswordBlocklist(path); err == nil {
			t.Fatal("LoadPasswordBlocklist() accepted oversized file")
		}
	})
	t.Run("oversized line", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "long-line.txt")
		if err := os.WriteFile(path, []byte(strings.Repeat("x", maximumBlocklistLineBytes+1)), 0o600); err != nil {
			t.Fatalf("write: %v", err)
		}
		if _, err := LoadPasswordBlocklist(path); err == nil {
			t.Fatal("LoadPasswordBlocklist() accepted oversized line")
		}
	})
	t.Run("exact line boundary", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "boundary-line.txt")
		line := strings.Repeat("x", maximumBlocklistLineBytes)
		if err := os.WriteFile(path, []byte(line), 0o600); err != nil {
			t.Fatalf("write: %v", err)
		}
		if _, err := LoadPasswordBlocklist(path); err != nil {
			t.Fatalf("LoadPasswordBlocklist() rejected exact boundary: %v", err)
		}
	})
}

func TestPasswordBlocklistScannerEnforcesActualReadLimit(t *testing.T) {
	t.Parallel()

	commentLine := strings.Repeat("#", maximumBlocklistLineBytes) + "\n"
	repeats := int(maximumBlocklistFileBytes/int64(len(commentLine))) + 2
	reader := strings.NewReader(strings.Repeat(commentLine, repeats))
	if _, err := scanPasswordBlocklist(reader); err == nil || !strings.Contains(err.Error(), "exceeds") {
		t.Fatalf("scanPasswordBlocklist() error = %v, want actual-byte limit rejection", err)
	}
}

func TestVersionedPasswordHashSupportsLongUnicodeAndLegacyVerification(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	password := strings.Repeat("界", 64) // 192 UTF-8 bytes, beyond bcrypt's input limit.
	hash, err := svc.HashPasswordForUser("person@example.com", password)
	if err != nil {
		t.Fatalf("HashPasswordForUser() error = %v", err)
	}
	if !strings.HasPrefix(hash, passwordHashPrefix) {
		t.Fatalf("hash = %q, want version prefix %q", hash, passwordHashPrefix)
	}
	if !svc.VerifyPassword(hash, password) {
		t.Fatal("versioned hash did not verify long Unicode password")
	}
	if svc.VerifyPassword(hash, password+"x") {
		t.Fatal("versioned hash accepted wrong password")
	}

	legacyPassword := "legacy-password-value"
	legacyHash, err := bcrypt.GenerateFromPassword([]byte(legacyPassword), passwordHashCost)
	if err != nil {
		t.Fatalf("legacy bcrypt: %v", err)
	}
	if !svc.VerifyPassword(string(legacyHash), legacyPassword) {
		t.Fatal("legacy bcrypt hash should remain verifiable")
	}
}

func TestLegacyVerificationDoesNotShortCircuitPasswordsOverBcryptHashLimit(t *testing.T) {
	t.Parallel()

	legacyHash, err := bcrypt.GenerateFromPassword([]byte("legacy-password-value"), bcrypt.MinCost)
	if err != nil {
		t.Fatalf("legacy bcrypt: %v", err)
	}
	longWrongPassword := strings.Repeat("x", 73)
	compareErr := bcrypt.CompareHashAndPassword(legacyHash, []byte(longWrongPassword))
	if errors.Is(compareErr, bcrypt.ErrPasswordTooLong) {
		t.Fatal("bcrypt comparison rejected a long candidate before performing legacy verification work")
	}
	if compareErr == nil {
		t.Fatal("legacy bcrypt unexpectedly accepted the wrong password")
	}
	if verifyStoredPassword(string(legacyHash), longWrongPassword) {
		t.Fatal("legacy verification accepted a wrong password over 72 bytes")
	}
}

func TestDummyPasswordHashUsesTheProductionWorkFactor(t *testing.T) {
	t.Parallel()

	cost, err := bcrypt.Cost([]byte(strings.TrimPrefix(dummyPasswordHash, passwordHashPrefix)))
	if err != nil {
		t.Fatalf("dummy hash is not valid bcrypt: %v", err)
	}
	if cost != passwordHashCost {
		t.Fatalf("dummy hash cost = %d, want %d", cost, passwordHashCost)
	}
}

func TestSuccessfulLegacyLoginUpgradesStoredHash(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	password := "legacy-password-value"
	legacyHash, err := bcrypt.GenerateFromPassword([]byte(password), passwordHashCost)
	if err != nil {
		t.Fatalf("legacy bcrypt: %v", err)
	}
	user, err := svc.store.CreateUser(
		context.Background(),
		"legacy@example.com",
		"Legacy User",
		string(legacyHash),
		RoleRead,
		defaultUserStatusActive,
	)
	if err != nil {
		t.Fatalf("CreateUser() error = %v", err)
	}

	if _, _, loginErr := svc.Login(context.Background(), user.Email, password); loginErr != nil {
		t.Fatalf("Login() error = %v", loginErr)
	}
	_, upgradedHash, err := svc.store.GetUserWithPasswordHash(context.Background(), user.ID)
	if err != nil {
		t.Fatalf("GetUserWithPasswordHash() error = %v", err)
	}
	if !strings.HasPrefix(upgradedHash, passwordHashPrefix) {
		t.Fatalf("stored hash = %q, want versioned upgrade", upgradedHash)
	}
}
