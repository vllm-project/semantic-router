package accesscontrol

import (
	"strings"
	"time"
)

type SubjectKind string

const (
	SubjectKindUser   SubjectKind = "user"
	SubjectKindTeam   SubjectKind = "team"
	SubjectKindAPIKey SubjectKind = "api_key"
)

func (k SubjectKind) Valid() bool {
	return k == SubjectKindUser || k == SubjectKindTeam || k == SubjectKindAPIKey
}

type SubjectRef struct {
	NamespaceID NamespaceID
	ID          SubjectID
	Kind        SubjectKind
}

func (s SubjectRef) Validate() error {
	var kindErr error
	if !s.Kind.Valid() {
		kindErr = invalid("kind", "is not a valid subject kind")
	}
	return joinValidation(
		validateRequired("namespace_id", string(s.NamespaceID)),
		validateRequired("id", string(s.ID)),
		kindErr,
	)
}

type Subject struct {
	NamespaceID NamespaceID
	ID          SubjectID
	Kind        SubjectKind
}

func (s Subject) Ref() SubjectRef {
	return SubjectRef(s)
}

func (s Subject) Validate() error { return s.Ref().Validate() }

type User struct {
	NamespaceID NamespaceID
	ID          UserID
	Email       string
	DisplayName string
	Status      UserStatus
	CreatedAt   time.Time
	UpdatedAt   time.Time
}

func (u User) SubjectRef() SubjectRef {
	return SubjectRef{NamespaceID: u.NamespaceID, ID: SubjectID(u.ID), Kind: SubjectKindUser}
}

func (u User) Validate() error {
	var statusErr error
	if !u.Status.Valid() {
		statusErr = invalid("status", "is not a valid user status")
	}
	return joinValidation(
		validateRequired("namespace_id", string(u.NamespaceID)),
		validateRequired("id", string(u.ID)),
		validateNormalizedEmail(u.Email),
		validateRequired("display_name", u.DisplayName),
		statusErr,
		validateTimestamps(u.CreatedAt, u.UpdatedAt),
	)
}

// NormalizeEmail applies the canonical normalization owned by the domain. It
// deliberately does not claim to validate mailbox deliverability.
func NormalizeEmail(email string) string {
	return strings.ToLower(strings.TrimSpace(email))
}

func validateNormalizedEmail(email string) error {
	if err := validateRequired("email", email); err != nil {
		return err
	}
	if NormalizeEmail(email) != email {
		return invalid("email", "must be normalized to lowercase without surrounding whitespace")
	}
	local, domain, ok := strings.Cut(email, "@")
	if !ok || local == "" || domain == "" || strings.Contains(domain, "@") {
		return invalid("email", "must contain one non-empty local and domain part")
	}
	return nil
}

type Team struct {
	NamespaceID NamespaceID
	ID          TeamID
	Name        string
	Status      TeamStatus
	CreatedAt   time.Time
	UpdatedAt   time.Time
}

func (t Team) SubjectRef() SubjectRef {
	return SubjectRef{NamespaceID: t.NamespaceID, ID: SubjectID(t.ID), Kind: SubjectKindTeam}
}

func (t Team) Validate() error {
	var statusErr error
	if !t.Status.Valid() {
		statusErr = invalid("status", "is not a valid team status")
	}
	return joinValidation(
		validateRequired("namespace_id", string(t.NamespaceID)),
		validateRequired("id", string(t.ID)),
		validateRequired("name", t.Name),
		statusErr,
		validateTimestamps(t.CreatedAt, t.UpdatedAt),
	)
}

type TeamRole string

const (
	TeamRoleMember TeamRole = "member"
	TeamRoleAdmin  TeamRole = "admin"
)

func (r TeamRole) Valid() bool { return r == TeamRoleMember || r == TeamRoleAdmin }

type TeamMembership struct {
	NamespaceID NamespaceID
	TeamID      TeamID
	UserID      UserID
	Role        TeamRole
	Status      MembershipStatus
	CreatedAt   time.Time
	UpdatedAt   time.Time
}

func (m TeamMembership) Validate() error {
	var roleErr, statusErr error
	if !m.Role.Valid() {
		roleErr = invalid("role", "is not a valid team role")
	}
	if !m.Status.Valid() {
		statusErr = invalid("status", "is not a valid membership status")
	}
	return joinValidation(
		validateRequired("namespace_id", string(m.NamespaceID)),
		validateRequired("team_id", string(m.TeamID)),
		validateRequired("user_id", string(m.UserID)),
		roleErr,
		statusErr,
		validateTimestamps(m.CreatedAt, m.UpdatedAt),
	)
}
