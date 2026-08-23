package accesscontrol

// APIKeyRelationships contains authoritative resources needed to validate key
// ownership. It is deliberately supplied by the application service rather
// than loaded by the domain package.
type APIKeyRelationships struct {
	OwnerUser         *User
	OwnerTeam         *Team
	ContextTeam       *Team
	ContextMembership *TeamMembership
}

// ValidateAPIKeyRelationships enforces cross-resource ownership and context
// invariants before a create or reassignment transaction is committed.
func ValidateAPIKeyRelationships(key APIKey, relationships APIKeyRelationships) error {
	if err := key.Validate(); err != nil {
		return err
	}

	switch key.Owner.Kind {
	case SubjectKindUser:
		return validateUserOwnedKey(key, relationships)
	case SubjectKindTeam:
		return validateTeamOwnedKey(key, relationships)
	default:
		return invalid("owner", "must be a user or team")
	}
}

func validateUserOwnedKey(key APIKey, relationships APIKeyRelationships) error {
	if err := validateUserOwner(key, relationships); err != nil {
		return err
	}
	return validateUserKeyContext(key, relationships)
}

func validateUserOwner(key APIKey, relationships APIKeyRelationships) error {
	if relationships.OwnerUser == nil {
		return invalid("owner_user", "must be supplied for a user-owned key")
	}
	if relationships.OwnerTeam != nil {
		return invalid("owner_team", "must be empty for a user-owned key")
	}
	owner := relationships.OwnerUser
	if owner.NamespaceID != key.NamespaceID || SubjectID(owner.ID) != key.Owner.ID {
		return invalid("owner_user", "does not match the key owner in its namespace")
	}
	if owner.Status != UserStatusActive {
		return invalid("owner_user", "must be active")
	}
	return nil
}

func validateUserKeyContext(key APIKey, relationships APIKeyRelationships) error {
	if key.ContextTeamID == "" {
		if relationships.ContextTeam != nil || relationships.ContextMembership != nil {
			return invalid("context_team", "must be empty when context_team_id is not set")
		}
		return nil
	}
	if relationships.ContextTeam == nil || relationships.ContextMembership == nil {
		return invalid("context_team", "requires an active team and membership")
	}
	return validateContextMembership(key, *relationships.OwnerUser, *relationships.ContextTeam, *relationships.ContextMembership)
}

func validateContextMembership(key APIKey, owner User, team Team, membership TeamMembership) error {
	if team.NamespaceID != key.NamespaceID || team.ID != key.ContextTeamID {
		return invalid("context_team", "does not match context_team_id in the key namespace")
	}
	if team.Status != TeamStatusActive {
		return invalid("context_team", "must be active")
	}
	if membership.NamespaceID != key.NamespaceID || membership.TeamID != team.ID || membership.UserID != owner.ID {
		return invalid("context_membership", "must link the owner user to the context team")
	}
	if membership.Status != MembershipStatusActive {
		return invalid("context_membership", "must be active")
	}
	return nil
}

func validateTeamOwnedKey(key APIKey, relationships APIKeyRelationships) error {
	if relationships.OwnerTeam == nil {
		return invalid("owner_team", "must be supplied for a team-owned key")
	}
	if relationships.OwnerUser != nil {
		return invalid("owner_user", "must be empty for a team-owned key")
	}
	if relationships.ContextTeam != nil || relationships.ContextMembership != nil {
		return invalid("context_team", "is derived from the owner for a team-owned key")
	}
	owner := relationships.OwnerTeam
	if owner.NamespaceID != key.NamespaceID || SubjectID(owner.ID) != key.Owner.ID {
		return invalid("owner_team", "does not match the key owner in its namespace")
	}
	if owner.Status != TeamStatusActive {
		return invalid("owner_team", "must be active")
	}
	return nil
}
