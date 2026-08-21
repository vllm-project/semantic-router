package accesscontrol

import (
	"context"
	"strings"
)

func normalizeTeamMemberships(teamID string, memberships []TeamMembership) ([]TeamMembership, error) {
	result := make([]TeamMembership, 0, len(memberships))
	seen := make(map[string]struct{}, len(memberships))
	for _, membership := range memberships {
		membership.UserID = strings.TrimSpace(membership.UserID)
		membership.Role = strings.ToLower(strings.TrimSpace(membership.Role))
		if membership.UserID == "" {
			return nil, validationError("Team member is required")
		}
		if _, exists := seen[membership.UserID]; exists {
			return nil, validationError("each user can appear only once in a Team")
		}
		if membership.Role != TeamRoleAdmin && membership.Role != TeamRoleMember {
			return nil, validationError("Team role must be admin or member")
		}
		seen[membership.UserID] = struct{}{}
		membership.TeamID = teamID
		result = append(result, membership)
	}
	return result, nil
}

func (s *Service) validateAssignableBudget(ctx context.Context, budgetID string) error {
	budget, err := s.store.GetBudget(ctx, budgetID)
	if err != nil {
		return validationError("selected budget does not exist")
	}
	if !budget.Enabled {
		return validationError("select an active budget")
	}
	return nil
}

func (s *Service) validateBudgetMutation(ctx context.Context, item Budget) error {
	if item.ID == "" || item.Enabled {
		return nil
	}
	current, err := s.store.GetBudget(ctx, item.ID)
	if err != nil {
		return err
	}
	if current.AssignmentCount > 0 {
		return validationError("remove this budget from its assignments before disabling it")
	}
	return nil
}
