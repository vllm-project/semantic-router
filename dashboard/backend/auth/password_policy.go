package auth

import (
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"errors"
	"fmt"
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
	passwordHashPrefix         = "$vsr$bcrypt-sha256$v1$"
	bcryptMaximumPasswordBytes = 72
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

// hashPasswordForStorage preserves rollback compatibility for ordinary
// already-normalized bcrypt inputs. Passwords that exceed bcrypt's 72-byte
// boundary, or whose Unicode normalization changes their byte sequence, use
// the versioned SHA-256 prehash format so their complete normalized value is
// authenticated instead of truncated.
func hashPasswordForStorage(originalPassword, normalizedPassword string) (string, error) {
	if originalPassword == normalizedPassword && len([]byte(normalizedPassword)) <= bcryptMaximumPasswordBytes {
		hash, err := bcrypt.GenerateFromPassword([]byte(normalizedPassword), passwordHashCost)
		return string(hash), err
	}
	return hashVersionedPassword(normalizedPassword)
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
	// Legacy rows contain a plain bcrypt hash. Keep them byte-stable during the
	// dual-read release so the ordinary-password path remains rollback-safe.
	return bcrypt.CompareHashAndPassword([]byte(hash), []byte(password)) == nil
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
