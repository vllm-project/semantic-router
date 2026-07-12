package auth

import (
	"fmt"
	"strings"
	"unicode"
	"unicode/utf8"
)

const minimumJWTSecretBytes = 32

const jwtSecretGenerationAdvice = "use a CSPRNG, for example: openssl rand -base64 32"

func validateConfiguredJWTSecret(secret string) error {
	if !utf8.ValidString(secret) {
		return invalidConfiguredJWTSecret("must contain valid UTF-8 text")
	}
	if strings.TrimSpace(secret) != secret {
		return invalidConfiguredJWTSecret("must not have leading or trailing whitespace")
	}
	for _, r := range secret {
		if unicode.IsControl(r) {
			return invalidConfiguredJWTSecret("must not contain control characters")
		}
	}
	if len(secret) < minimumJWTSecretBytes {
		return invalidConfiguredJWTSecret(fmt.Sprintf(
			"must be at least %d bytes",
			minimumJWTSecretBytes,
		))
	}
	if isObviousJWTSecretPlaceholder(secret) {
		return invalidConfiguredJWTSecret("must not be a repeated or known placeholder value")
	}
	return nil
}

func invalidConfiguredJWTSecret(reason string) error {
	return fmt.Errorf("configured JWT signing secret %s; %s", reason, jwtSecretGenerationAdvice)
}

func isObviousJWTSecretPlaceholder(secret string) bool {
	var first rune
	allSame := true
	for index, r := range secret {
		if index == 0 {
			first = r
			continue
		}
		if r != first {
			allSame = false
		}
	}
	if allSame {
		return true
	}

	var canonical strings.Builder
	for _, r := range strings.ToLower(secret) {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			canonical.WriteRune(r)
		}
	}
	value := canonical.String()
	for _, fragment := range []string{
		"defaultjwtsecret",
		"examplejwtsecret",
		"placeholder",
		"replacewithrandom",
		"testjwtsecret",
		"yourjwtsecret",
	} {
		if strings.Contains(value, fragment) {
			return true
		}
	}
	for _, token := range []string{"changeme", "jwtsecret", "password", "secret"} {
		if len(value) >= minimumJWTSecretBytes &&
			len(value)%len(token) == 0 &&
			strings.Repeat(token, len(value)/len(token)) == value {
			return true
		}
	}
	return false
}
