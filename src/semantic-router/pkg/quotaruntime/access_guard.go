package quotaruntime

import "fmt"

// AdmissionCheckKind is a deliberately small assertion language evaluated by
// the same Lua operation that consumes quota. It lets AccessRuntime pin the
// credential verification revision and recheck every security projection
// without a read/check/write gap.
type AdmissionCheckKind string

const (
	AdmissionCheckHashEqual    AdmissionCheckKind = "hash_equal"
	AdmissionCheckStringEqual  AdmissionCheckKind = "string_equal"
	AdmissionCheckKeyAbsent    AdmissionCheckKind = "key_absent"
	AdmissionCheckSetMember    AdmissionCheckKind = "set_member"
	AdmissionCheckNotBefore    AdmissionCheckKind = "hash_not_before"
	AdmissionCheckExpiresAfter AdmissionCheckKind = "hash_expires_after"
)

// AdmissionPrecondition describes one partition-local access projection
// assertion. Failure must be a public non-success outcome; Reason is a stable
// machine-facing code rather than a secret-bearing store value.
type AdmissionPrecondition struct {
	Key      string
	Kind     AdmissionCheckKind
	Field    string
	Expected string
	Failure  AdmissionDisposition
	Reason   string
}

func (p AdmissionPrecondition) Validate() error {
	if err := validateOpaque("precondition key", p.Key); err != nil {
		return err
	}
	switch p.Failure {
	case AdmissionUnauthenticated, AdmissionForbidden, AdmissionUnavailable:
	default:
		return fmt.Errorf("%w: invalid precondition failure %q", ErrInvalidRequest, p.Failure)
	}
	if err := validateOpaque("precondition reason", p.Reason); err != nil {
		return err
	}
	if len(p.Reason) > 128 {
		return fmt.Errorf("%w: precondition reason is too long", ErrInvalidRequest)
	}
	switch p.Kind {
	case AdmissionCheckHashEqual:
		if err := validateOpaque("precondition field", p.Field); err != nil {
			return err
		}
		if err := validateProjectionValue(p.Expected); err != nil {
			return err
		}
	case AdmissionCheckStringEqual, AdmissionCheckSetMember:
		if p.Field != "" {
			return fmt.Errorf("%w: %s does not accept a field", ErrInvalidRequest, p.Kind)
		}
		if err := validateProjectionValue(p.Expected); err != nil {
			return err
		}
	case AdmissionCheckKeyAbsent:
		if p.Field != "" || p.Expected != "" {
			return fmt.Errorf("%w: key_absent accepts neither field nor expected value", ErrInvalidRequest)
		}
	case AdmissionCheckNotBefore, AdmissionCheckExpiresAfter:
		if err := validateOpaque("precondition field", p.Field); err != nil {
			return err
		}
		if p.Expected != "" {
			return fmt.Errorf("%w: %s does not accept an expected value", ErrInvalidRequest, p.Kind)
		}
	default:
		return fmt.Errorf("%w: unsupported precondition kind %q", ErrInvalidRequest, p.Kind)
	}
	return nil
}

func validateProjectionValue(value string) error {
	if value == "" || len(value) > 1024 {
		return fmt.Errorf("%w: expected projection value is empty or too long", ErrInvalidRequest)
	}
	for index := range value {
		if value[index] == 0 {
			return fmt.Errorf("%w: expected projection value contains NUL", ErrInvalidRequest)
		}
	}
	return nil
}

// AccessProjectionKeyspace defines the partition-local keys that can
// participate in atomic admission. CredentialDirectoryKey is intentionally
// global and read-only: it may locate a partition, but must never be passed as
// an AdmissionPrecondition.
type AccessProjectionKeyspace struct {
	tag    string
	prefix string
}

func NewAccessProjectionKeyspace(partition string) (AccessProjectionKeyspace, error) {
	return NewAccessProjectionKeyspaceWithPrefix("", partition)
}

func NewAccessProjectionKeyspaceWithPrefix(prefix, partition string) (AccessProjectionKeyspace, error) {
	keys, err := newPartitionKeysWithPrefix(prefix, partition)
	if err != nil {
		return AccessProjectionKeyspace{}, err
	}
	return AccessProjectionKeyspace{tag: keys.tag, prefix: prefix}, nil
}

func CredentialDirectoryKey(kind, publicID string) (string, error) {
	return CredentialDirectoryKeyWithPrefix("", kind, publicID)
}

func CredentialDirectoryKeyWithPrefix(prefix, kind, publicID string) (string, error) {
	if err := validateKeyPrefix(prefix); err != nil {
		return "", err
	}
	if err := validateOpaque("credential kind", kind); err != nil {
		return "", err
	}
	if err := validateOpaque("credential public ID", publicID); err != nil {
		return "", err
	}
	return prefixedKey(
		prefix,
		"access:credential-directory:"+keyComponent(kind)+":"+keyComponent(publicID),
	), nil
}

func (k AccessProjectionKeyspace) Credential(kind, publicID string) string {
	return prefixedKey(k.prefix, "access:"+k.tag+":credential:"+keyComponent(kind)+":"+keyComponent(publicID))
}

func (k AccessProjectionKeyspace) LogicalKey(keyID string) string {
	return prefixedKey(k.prefix, "access:"+k.tag+":key:"+keyComponent(keyID))
}

func (k AccessProjectionKeyspace) Active(keyID string) string {
	return prefixedKey(k.prefix, "access:"+k.tag+":active:"+keyComponent(keyID))
}

func (k AccessProjectionKeyspace) Policy(keyID, revision string) string {
	return prefixedKey(k.prefix, "access:"+k.tag+":policy:"+keyComponent(keyID)+":"+keyComponent(revision))
}

func (k AccessProjectionKeyspace) Deny(kind, resourceID string) string {
	return prefixedKey(k.prefix, "access:"+k.tag+":deny:"+keyComponent(kind)+":"+keyComponent(resourceID))
}

func (k AccessProjectionKeyspace) ManagementSession(sessionID string) string {
	return prefixedKey(k.prefix, "management:"+k.tag+":session:"+keyComponent(sessionID))
}
