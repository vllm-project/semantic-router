package accesscontrol

type NamespaceStatus string

const (
	NamespaceStatusActive   NamespaceStatus = "active"
	NamespaceStatusDisabled NamespaceStatus = "disabled"
)

func (s NamespaceStatus) Valid() bool {
	return s == NamespaceStatusActive || s == NamespaceStatusDisabled
}

type UserStatus string

const (
	UserStatusActive   UserStatus = "active"
	UserStatusDisabled UserStatus = "disabled"
	UserStatusDeleted  UserStatus = "deleted"
)

func (s UserStatus) Valid() bool {
	return s == UserStatusActive || s == UserStatusDisabled || s == UserStatusDeleted
}

type TeamStatus string

const (
	TeamStatusActive   TeamStatus = "active"
	TeamStatusDisabled TeamStatus = "disabled"
)

func (s TeamStatus) Valid() bool {
	return s == TeamStatusActive || s == TeamStatusDisabled
}

type MembershipStatus string

const (
	MembershipStatusActive   MembershipStatus = "active"
	MembershipStatusDisabled MembershipStatus = "disabled"
)

func (s MembershipStatus) Valid() bool {
	return s == MembershipStatusActive || s == MembershipStatusDisabled
}

type APIKeyStatus string

const (
	APIKeyStatusActive   APIKeyStatus = "active"
	APIKeyStatusDisabled APIKeyStatus = "disabled"
	APIKeyStatusDeleted  APIKeyStatus = "deleted"
)

func (s APIKeyStatus) Valid() bool {
	return s == APIKeyStatusActive || s == APIKeyStatusDisabled || s == APIKeyStatusDeleted
}

type CredentialStatus string

const (
	CredentialStatusActive   CredentialStatus = "active"
	CredentialStatusRetiring CredentialStatus = "retiring"
	CredentialStatusRevoked  CredentialStatus = "revoked"
	CredentialStatusExpired  CredentialStatus = "expired"
)

func (s CredentialStatus) Valid() bool {
	switch s {
	case CredentialStatusActive, CredentialStatusRetiring, CredentialStatusRevoked, CredentialStatusExpired:
		return true
	default:
		return false
	}
}

type PolicyStatus string

const (
	PolicyStatusDraft    PolicyStatus = "draft"
	PolicyStatusActive   PolicyStatus = "active"
	PolicyStatusDisabled PolicyStatus = "disabled"
)

func (s PolicyStatus) Valid() bool {
	return s == PolicyStatusDraft || s == PolicyStatusActive || s == PolicyStatusDisabled
}

type BindingStatus string

const (
	BindingStatusActive   BindingStatus = "active"
	BindingStatusDisabled BindingStatus = "disabled"
)

func (s BindingStatus) Valid() bool {
	return s == BindingStatusActive || s == BindingStatusDisabled
}

type PrincipalStatus string

const (
	PrincipalStatusActive   PrincipalStatus = "active"
	PrincipalStatusDisabled PrincipalStatus = "disabled"
)

func (s PrincipalStatus) Valid() bool {
	return s == PrincipalStatusActive || s == PrincipalStatusDisabled
}

type RoleStatus string

const (
	RoleStatusActive   RoleStatus = "active"
	RoleStatusDisabled RoleStatus = "disabled"
)

func (s RoleStatus) Valid() bool {
	return s == RoleStatusActive || s == RoleStatusDisabled
}
