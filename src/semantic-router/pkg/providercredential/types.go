package providercredential

import (
	"errors"
	"fmt"
	"net"
	"net/url"
	"path"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscredential"
)

type Status string

// Mode is the immutable credential requirement selected by the active
// Provider definition when a credential is created. It deliberately excludes
// "none": Providers that do not accept credentials cannot own this resource.
type Mode string

const (
	ModeOptional Mode = "optional"
	ModeRequired Mode = "required"
)

const (
	StatusActive   Status = "active"
	StatusDisabled Status = "disabled"
	StatusDeleted  Status = "deleted"
)

type VersionStatus string

const (
	VersionActive   VersionStatus = "active"
	VersionRetiring VersionStatus = "retiring"
	VersionRevoked  VersionStatus = "revoked"
)

var (
	providerIDPattern      = regexp.MustCompile(`^[a-z][a-z0-9._-]{0,127}$`)
	catalogRevisionPattern = regexp.MustCompile(`^sha256:[a-f0-9]{64}$`)
)

// Credential binds one immutable provider definition, credential adapter, and
// normalized origin. Secret rotation changes ActiveVersionID, never a binding.
type Credential struct {
	ID                  string
	NamespaceID         string
	Name                string
	ProviderID          string
	CredentialMode      Mode
	CredentialAdapterID string
	CatalogRevision     string
	NormalizedOrigin    string
	Status              Status
	ActiveVersionID     *string
	Revision            uint64
	CreatedAt           time.Time
	UpdatedAt           time.Time
	DeletedAt           *time.Time
}

// Version contains only envelope-encrypted secret material and lifecycle
// metadata. Ciphertext, nonce, and key version are never management response
// fields.
type Version struct {
	ID           string
	NamespaceID  string
	CredentialID string
	Envelope     accesscredential.Envelope
	Status       VersionStatus
	NotBefore    time.Time
	ExpiresAt    *time.Time
	RevokedAt    *time.Time
	CreatedAt    time.Time
}

func (c Credential) Validate() error {
	if err := validateUUIDs(map[string]string{"id": c.ID, "namespace_id": c.NamespaceID}); err != nil {
		return err
	}
	if err := ValidateName(c.Name); err != nil {
		return err
	}
	if !providerIDPattern.MatchString(c.ProviderID) {
		return errors.New("provider id is invalid")
	}
	if c.CredentialMode != ModeOptional && c.CredentialMode != ModeRequired {
		return errors.New("provider credential mode is invalid")
	}
	if !providerIDPattern.MatchString(c.CredentialAdapterID) {
		return errors.New("provider credential adapter id is invalid")
	}
	if !catalogRevisionPattern.MatchString(c.CatalogRevision) {
		return errors.New("provider catalog revision is invalid")
	}
	normalized, err := NormalizeOrigin(c.NormalizedOrigin)
	if err != nil || normalized != c.NormalizedOrigin {
		return errors.New("provider credential origin is not canonical")
	}
	if c.Revision == 0 || c.CreatedAt.IsZero() || c.UpdatedAt.IsZero() || c.UpdatedAt.Before(c.CreatedAt) {
		return errors.New("provider credential revision or timestamps are invalid")
	}
	switch c.Status {
	case StatusActive:
		if c.ActiveVersionID == nil || c.DeletedAt != nil {
			return errors.New("active provider credential requires an active version and no deletion time")
		}
	case StatusDisabled:
		if c.ActiveVersionID != nil || c.DeletedAt != nil {
			return errors.New("disabled provider credential must clear its active version")
		}
	case StatusDeleted:
		if c.ActiveVersionID != nil || c.DeletedAt == nil || c.DeletedAt.Before(c.CreatedAt) {
			return errors.New("deleted provider credential must clear its active version and record deletion")
		}
	default:
		return errors.New("provider credential status is invalid")
	}
	if c.ActiveVersionID != nil {
		if _, err := uuid.Parse(*c.ActiveVersionID); err != nil {
			return errors.New("active provider credential version id must be a UUID")
		}
	}
	return nil
}

// ValidateName applies the same canonical display-name contract without
// requiring callers to construct a synthetic credential.
func ValidateName(name string) error {
	if !canonicalText(name, 1, 256) {
		return errors.New("provider credential name must be canonical and at most 256 bytes")
	}
	return nil
}

// ValidateProviderID validates catalog Provider and credential-adapter IDs
// without requiring callers to construct synthetic credential metadata.
func ValidateProviderID(value string) error {
	if !providerIDPattern.MatchString(value) {
		return errors.New("provider id is invalid")
	}
	return nil
}

func (v Version) Validate() error {
	if err := validateUUIDs(map[string]string{
		"id": v.ID, "namespace_id": v.NamespaceID, "credential_id": v.CredentialID,
	}); err != nil {
		return err
	}
	hasCompleteEnvelope := v.Envelope.KeyVersion != "" && len(v.Envelope.Nonce) > 0 && len(v.Envelope.Ciphertext) > 0
	hasAnyEnvelope := v.Envelope.KeyVersion != "" || len(v.Envelope.Nonce) > 0 || len(v.Envelope.Ciphertext) > 0
	if v.NotBefore.IsZero() || v.CreatedAt.IsZero() {
		return errors.New("provider credential version times are invalid")
	}
	if v.ExpiresAt != nil && !v.ExpiresAt.After(v.NotBefore) {
		return errors.New("provider credential version expiry must follow not-before")
	}
	switch v.Status {
	case VersionActive:
		if v.RevokedAt != nil || !hasCompleteEnvelope {
			return errors.New("active provider credential version requires an encrypted envelope and cannot be revoked")
		}
	case VersionRetiring:
		if v.ExpiresAt == nil || v.RevokedAt != nil || !hasCompleteEnvelope {
			return errors.New("retiring provider credential version requires an encrypted envelope and bounded expiry")
		}
	case VersionRevoked:
		if v.RevokedAt == nil || v.RevokedAt.Before(v.CreatedAt) || hasAnyEnvelope {
			return errors.New("revoked provider credential version requires revocation time and erased secret material")
		}
	default:
		return errors.New("provider credential version status is invalid")
	}
	return nil
}

// NormalizeOrigin returns scheme://host[:port][/base-path]. Origins never
// contain credentials, query, fragment, dot segments, or encoded separators.
// Network egress policy is a separate mandatory check after DNS resolution.
func NormalizeOrigin(raw string) (string, error) {
	if raw == "" || strings.TrimSpace(raw) != raw || strings.Contains(raw, "\\") {
		return "", errors.New("origin is empty or non-canonical")
	}
	parsed, err := url.Parse(raw)
	if err != nil || parsed.Opaque != "" || parsed.User != nil || parsed.RawQuery != "" || parsed.Fragment != "" {
		return "", errors.New("origin contains forbidden URL components")
	}
	scheme := strings.ToLower(parsed.Scheme)
	if scheme != "https" && scheme != "http" {
		return "", errors.New("origin scheme must be http or https")
	}
	hostname := strings.ToLower(parsed.Hostname())
	if hostname == "" || !asciiHostname(hostname) {
		return "", errors.New("origin hostname is invalid")
	}
	port := parsed.Port()
	if port != "" {
		value, parseErr := strconv.ParseUint(port, 10, 16)
		if parseErr != nil || value == 0 {
			return "", errors.New("origin port is invalid")
		}
		if (scheme == "https" && value == 443) || (scheme == "http" && value == 80) {
			port = ""
		}
	}
	host := hostname
	if strings.Contains(hostname, ":") {
		host = "[" + hostname + "]"
	}
	if port != "" {
		host = net.JoinHostPort(hostname, port)
	}
	basePath, err := normalizeBasePath(parsed.EscapedPath())
	if err != nil {
		return "", err
	}
	return scheme + "://" + host + basePath, nil
}

func normalizeBasePath(escaped string) (string, error) {
	if escaped == "" || escaped == "/" {
		return "", nil
	}
	lower := strings.ToLower(escaped)
	if strings.Contains(lower, "%2f") || strings.Contains(lower, "%5c") {
		return "", errors.New("origin path contains an encoded separator")
	}
	decoded, err := url.PathUnescape(escaped)
	if err != nil || !strings.HasPrefix(decoded, "/") {
		return "", errors.New("origin path is invalid")
	}
	for _, segment := range strings.Split(decoded, "/") {
		if segment == "." || segment == ".." {
			return "", errors.New("origin path contains a dot segment")
		}
	}
	cleaned := strings.TrimSuffix(path.Clean(decoded), "/")
	if cleaned == "." || cleaned == "/" {
		return "", nil
	}
	return (&url.URL{Path: cleaned}).EscapedPath(), nil
}

func asciiHostname(host string) bool {
	if parsed := net.ParseIP(host); parsed != nil {
		return parsed.String() == host || strings.EqualFold(parsed.String(), host)
	}
	for _, char := range host {
		if char > 127 || (char != '.' && char != '-' && (char < 'a' || char > 'z') && (char < '0' || char > '9')) {
			return false
		}
	}
	return !strings.HasPrefix(host, ".") && !strings.HasSuffix(host, ".") && !strings.Contains(host, "..")
}

func validateUUIDs(values map[string]string) error {
	for field, value := range values {
		if _, err := uuid.Parse(value); err != nil {
			return fmt.Errorf("%s must be a UUID", field)
		}
	}
	return nil
}

func canonicalText(value string, minimum, maximum int) bool {
	return len(value) >= minimum && len(value) <= maximum && strings.TrimSpace(value) == value &&
		!strings.ContainsAny(value, "\x00\r\n\t")
}
