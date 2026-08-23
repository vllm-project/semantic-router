package accesscontrol

import "time"

var testTime = time.Date(2026, 8, 22, 12, 0, 0, 0, time.UTC)

func validUser() User {
	return User{
		NamespaceID: "ns-1",
		ID:          "user-1",
		Email:       "user@example.com",
		DisplayName: "User One",
		Status:      UserStatusActive,
		CreatedAt:   testTime,
		UpdatedAt:   testTime,
	}
}

func validTeam() Team {
	return Team{
		NamespaceID: "ns-1",
		ID:          "team-1",
		Name:        "Team One",
		Status:      TeamStatusActive,
		CreatedAt:   testTime,
		UpdatedAt:   testTime,
	}
}

func validMembership() TeamMembership {
	return TeamMembership{
		NamespaceID: "ns-1",
		TeamID:      "team-1",
		UserID:      "user-1",
		Role:        TeamRoleMember,
		Status:      MembershipStatusActive,
		CreatedAt:   testTime,
		UpdatedAt:   testTime,
	}
}

func validUserKey() APIKey {
	return APIKey{
		NamespaceID:     "ns-1",
		ID:              "key-1",
		Name:            "User key",
		Owner:           validUser().SubjectRef(),
		Status:          APIKeyStatusActive,
		PolicyEpoch:     1,
		DelegationEpoch: 1,
		Revision:        1,
		CreatedAt:       testTime,
		UpdatedAt:       testTime,
	}
}

func validAccessBinding(id string, kind SubjectKind, subjectID string) AccessPolicyBinding {
	return AccessPolicyBinding{
		ID:          PolicyBindingID(id),
		NamespaceID: "ns-1",
		Subject: SubjectRef{
			NamespaceID: "ns-1",
			ID:          SubjectID(subjectID),
			Kind:        kind,
		},
		PolicyID: "access-1",
		Status:   BindingStatusActive,
		Revision: 1,
	}
}

func validRateBinding(id string, kind SubjectKind, subjectID string, mode RateBindingMode) RateLimitBinding {
	return RateLimitBinding{
		ID:               PolicyBindingID(id),
		NamespaceID:      "ns-1",
		Subject:          SubjectRef{NamespaceID: "ns-1", ID: SubjectID(subjectID), Kind: kind},
		PolicyID:         "rate-1",
		Mode:             mode,
		QuotaPartitionID: "partition-1",
		Status:           BindingStatusActive,
		Revision:         1,
	}
}

func validRoleBinding(role ManagementRole, scope Scope) ManagementRoleBinding {
	return ManagementRoleBinding{
		ID:                "role-binding-1",
		PrincipalID:       "principal-1",
		RoleID:            role.ID,
		Scope:             scope,
		DelegationCeiling: DelegablePermissions(),
		Status:            BindingStatusActive,
		Revision:          1,
	}
}
