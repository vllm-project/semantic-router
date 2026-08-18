package handlers

import (
	"net/http"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"
)

func summarizeModelVerificationText(value string, maxRunes int) string {
	fields := strings.FieldsFunc(strings.TrimSpace(value), func(character rune) bool {
		return unicode.IsSpace(character) || unicode.IsControl(character)
	})
	normalized := strings.Join(fields, " ")
	if utf8.RuneCountInString(normalized) <= maxRunes {
		return normalized
	}
	runes := []rune(normalized)
	return strings.TrimSpace(string(runes[:maxRunes])) + "…"
}

func redactModelVerificationSecrets(value string, secrets ...string) string {
	redacted := value
	for _, secret := range secrets {
		if secret = strings.TrimSpace(secret); secret != "" {
			redacted = strings.ReplaceAll(redacted, secret, "[REDACTED]")
		}
	}
	return redacted
}

func modelVerificationCredentialSecrets(target modelVerificationTarget) []string {
	seen := make(map[string]struct{})
	result := make([]string, 0, len(target.extraHeaders)+1)
	add := func(value string) {
		value = strings.TrimSpace(value)
		if value == "" {
			return
		}
		if _, found := seen[value]; found {
			return
		}
		seen[value] = struct{}{}
		result = append(result, value)
	}
	add(target.apiKey)
	// Extra headers are provider-defined and may carry credentials even when
	// their names are not conventional. Treat every value as sensitive when
	// sanitizing generated evidence. Upstream error bodies are never surfaced.
	for _, value := range target.extraHeaders {
		add(value)
		parts := strings.Fields(value)
		if len(parts) == 2 && (strings.EqualFold(parts[0], "bearer") || strings.EqualFold(parts[0], "basic")) {
			add(parts[1])
		}
	}
	sort.Slice(result, func(left, right int) bool {
		return len(result[left]) > len(result[right])
	})
	return result
}

func copyModelVerificationHeaders(headers map[string]string) map[string]string {
	if len(headers) == 0 {
		return nil
	}
	result := make(map[string]string, len(headers))
	for key, value := range headers {
		result[key] = value
	}
	return result
}

func modelVerificationHeaderAllowed(header string) bool {
	name := http.CanonicalHeaderKey(strings.TrimSpace(header))
	if name == "" {
		return false
	}
	switch name {
	case "Connection", "Content-Length", "Host", "Proxy-Authorization", "Transfer-Encoding", "Upgrade":
		return false
	default:
		return true
	}
}
