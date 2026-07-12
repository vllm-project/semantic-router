package auth

import (
	"bufio"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"unicode"
	"unicode/utf8"

	"golang.org/x/crypto/bcrypt"
	"golang.org/x/text/unicode/norm"
)

const (
	minimumPasswordCodePoints = 15
	maximumPasswordCodePoints = 1024
	passwordHashCost          = 12
	// #nosec G101 -- public on-disk hash format marker, not a credential.
	passwordHashPrefix        = "$vsr$bcrypt-sha256$v1$"
	maximumBlocklistFileBytes = 8 * 1024 * 1024
	maximumBlocklistLineBytes = 4096
	maximumBlocklistEntries   = 250_000
)

const (
	PasswordPolicyTooShort  = "too_short"
	PasswordPolicyTooLong   = "too_long"
	PasswordPolicyInvalid   = "invalid_unicode"
	PasswordPolicyBlocked   = "blocked"
	PasswordPolicyUnchanged = "unchanged"
)

// PasswordPolicyError is safe to return to a caller choosing a new password.
// It never includes the submitted password or the matching denylist value.
type PasswordPolicyError struct {
	Code    string
	Message string
}

func (e *PasswordPolicyError) Error() string { return e.Message }

// PasswordBlocklist is the policy seam for local or privacy-preserving
// compromised-password data. Implementations must compare the complete
// password, never log it, and must not send the raw value to another service.
type PasswordBlocklist interface {
	Blocked(normalizedPassword, accountEmail string) bool
}

// PasswordPolicy validates and normalizes newly established passwords.
type PasswordPolicy struct {
	blocklist PasswordBlocklist
}

func NewPasswordPolicy(blocklist PasswordBlocklist) *PasswordPolicy {
	if blocklist == nil {
		blocklist = newLocalPasswordBlocklist()
	}
	return &PasswordPolicy{blocklist: blocklist}
}

// Validate applies the single-factor password requirements before hashing.
// Length is measured in Unicode code points after NFC normalization. No
// character-class composition rules are imposed.
func (p *PasswordPolicy) Validate(accountEmail, password string) (string, error) {
	if !utf8.ValidString(password) {
		return "", &PasswordPolicyError{
			Code:    PasswordPolicyInvalid,
			Message: "password must contain valid Unicode text",
		}
	}

	normalized := norm.NFC.String(password)
	length := utf8.RuneCountInString(normalized)
	if length < minimumPasswordCodePoints {
		return "", &PasswordPolicyError{
			Code: PasswordPolicyTooShort,
			Message: fmt.Sprintf(
				"password must be at least %d characters",
				minimumPasswordCodePoints,
			),
		}
	}
	if length > maximumPasswordCodePoints {
		return "", &PasswordPolicyError{
			Code: PasswordPolicyTooLong,
			Message: fmt.Sprintf(
				"password must be at most %d characters",
				maximumPasswordCodePoints,
			),
		}
	}
	if p.blocklist.Blocked(normalized, accountEmail) {
		return "", &PasswordPolicyError{
			Code:    PasswordPolicyBlocked,
			Message: "password is commonly used, compromised, or specific to this account or service",
		}
	}
	return normalized, nil
}

type localPasswordBlocklist struct {
	exactValues     map[string]struct{}
	canonicalValues map[string]struct{}
}

func newLocalPasswordBlocklist(additionalValues ...string) *localPasswordBlocklist {
	// These are complete, commonly guessed or service-specific values. The
	// canonical comparison also catches separator and case variants without
	// imposing a general composition rule.
	values := []string{
		"123456789012345",
		"adminadminadmin",
		"administrator123",
		"changemechangeme",
		"correcthorsebatterystaple",
		"dashboardpassword",
		"iloveyouiloveyou",
		"letmeinletmein",
		"password123456",
		"passwordpassword",
		"qwertyuiopasdfgh",
		"semanticrouter",
		"semanticrouterpassword",
		"vllmdashboardpassword",
		"vllmsemanticrouter",
		"welcome123456789",
	}
	// Exact entries cover a bounded set of common complete values whose
	// separator-insensitive canonical form is empty. They are denylist values,
	// not a general requirement that passwords contain a particular class of
	// character.
	for _, repeated := range []string{" ", "!", ".", "-", "_", "*"} {
		values = append(values, strings.Repeat(repeated, minimumPasswordCodePoints))
	}

	exact := make(map[string]struct{}, len(values)+len(additionalValues))
	canonical := make(map[string]struct{}, len(values)+len(additionalValues))
	for _, value := range append(values, additionalValues...) {
		normalized := norm.NFC.String(value)
		exact[normalized] = struct{}{}
		if canonicalValue := canonicalPasswordValue(normalized); canonicalValue != "" {
			canonical[canonicalValue] = struct{}{}
		}
	}
	return &localPasswordBlocklist{
		exactValues:     exact,
		canonicalValues: canonical,
	}
}

// LoadPasswordBlocklist merges a bounded local newline-delimited blocklist
// with the built-in minimum. Exactly empty lines and lines whose first byte is
// # are ignored. Leading and trailing whitespace is otherwise significant so
// that complete whitespace-bearing passwords can be represented. Loading
// fails closed at startup; entries are never logged.
func LoadPasswordBlocklist(path string) (PasswordBlocklist, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return newLocalPasswordBlocklist(), nil
	}
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open password blocklist: %w", err)
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return nil, fmt.Errorf("stat password blocklist: %w", err)
	}
	if !info.Mode().IsRegular() {
		return nil, errors.New("password blocklist must be a regular file")
	}
	if info.Size() > maximumBlocklistFileBytes {
		return nil, fmt.Errorf("password blocklist exceeds %d bytes", maximumBlocklistFileBytes)
	}

	entries, err := scanPasswordBlocklist(file)
	if err != nil {
		return nil, err
	}
	if len(entries) == 0 {
		return nil, errors.New("password blocklist contains no usable entries")
	}
	return newLocalPasswordBlocklist(entries...), nil
}

func scanPasswordBlocklist(reader io.Reader) ([]string, error) {
	limited := &io.LimitedReader{R: reader, N: maximumBlocklistFileBytes + 1}
	entries := make([]string, 0, 1024)
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
		entries = append(entries, norm.NFC.String(entry))
		if len(entries) > maximumBlocklistEntries {
			return nil, fmt.Errorf("password blocklist exceeds %d entries", maximumBlocklistEntries)
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("read password blocklist: %w", err)
	}
	if limited.N == 0 {
		return nil, fmt.Errorf("password blocklist exceeds %d bytes", maximumBlocklistFileBytes)
	}
	return entries, nil
}

func (b *localPasswordBlocklist) Blocked(password, accountEmail string) bool {
	normalized := norm.NFC.String(password)
	if _, found := b.exactValues[normalized]; found {
		return true
	}
	candidate := canonicalPasswordValue(normalized)
	if candidate != "" {
		if _, found := b.canonicalValues[candidate]; found {
			return true
		}
	}

	email := strings.TrimSpace(strings.ToLower(accountEmail))
	localPart := email
	if at := strings.IndexByte(email, '@'); at >= 0 {
		localPart = email[:at]
	}
	for _, accountValue := range []string{email, localPart} {
		accountValue = canonicalPasswordValue(accountValue)
		if candidate != "" && accountValue != "" && candidate == accountValue {
			return true
		}
	}
	return false
}

func canonicalPasswordValue(value string) string {
	var builder strings.Builder
	for _, r := range norm.NFKC.String(strings.ToLower(value)) {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			builder.WriteRune(r)
		}
	}
	return builder.String()
}

func hashVersionedPassword(normalizedPassword string) (string, error) {
	normalizedPassword = norm.NFC.String(normalizedPassword)
	digest := sha256.Sum256([]byte(normalizedPassword))
	prehash := make([]byte, hex.EncodedLen(len(digest)))
	hex.Encode(prehash, digest[:])
	hash, err := bcrypt.GenerateFromPassword(prehash, passwordHashCost)
	if err != nil {
		return "", err
	}
	return passwordHashPrefix + string(hash), nil
}

func verifyStoredPassword(hash, password string) bool {
	if hash == "" || !utf8.ValidString(password) {
		return false
	}
	if strings.HasPrefix(hash, passwordHashPrefix) {
		normalized := norm.NFC.String(password)
		digest := sha256.Sum256([]byte(normalized))
		prehash := make([]byte, hex.EncodedLen(len(digest)))
		hex.Encode(prehash, digest[:])
		stored := strings.TrimPrefix(hash, passwordHashPrefix)
		return bcrypt.CompareHashAndPassword([]byte(stored), prehash) == nil
	}
	// Legacy rows contain a plain bcrypt hash. Keep verification during the
	// migration window; successful login upgrades the row to the versioned form.
	return bcrypt.CompareHashAndPassword([]byte(hash), []byte(password)) == nil
}

func passwordHashNeedsUpgrade(hash string) bool {
	return hash != "" && !strings.HasPrefix(hash, passwordHashPrefix)
}

func passwordsEquivalent(first, second string) bool {
	firstDigest := sha256.Sum256([]byte(norm.NFC.String(first)))
	secondDigest := sha256.Sum256([]byte(norm.NFC.String(second)))
	return subtle.ConstantTimeCompare(firstDigest[:], secondDigest[:]) == 1
}

func asPasswordPolicyError(err error) (*PasswordPolicyError, bool) {
	var policyErr *PasswordPolicyError
	if !errors.As(err, &policyErr) {
		return nil, false
	}
	return policyErr, true
}
