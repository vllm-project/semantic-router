package managementserver

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/invitationmanagement"
)

func TestInvitationResponseEncodesEmptyGrantCeilingAsArray(t *testing.T) {
	wire, err := json.Marshal(invitationDTO(invitationmanagement.Invitation{
		Snapshot: invitationmanagement.OnboardingSnapshot{
			RoleGrants: []invitationmanagement.RoleGrant{{}},
		},
	}))
	if err != nil {
		t.Fatal(err)
	}
	var payload struct {
		Onboarding struct {
			RoleGrants []struct {
				DelegationCeiling json.RawMessage `json:"delegationCeiling"`
			} `json:"roleGrants"`
		} `json:"onboarding"`
	}
	if err := json.Unmarshal(wire, &payload); err != nil {
		t.Fatal(err)
	}
	if string(payload.Onboarding.RoleGrants[0].DelegationCeiling) != "[]" {
		t.Fatalf("InvitationRoleGrant.delegationCeiling must be an array: %s", wire)
	}
}
