package postgres

import (
	"fmt"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/policybulk"
)

func TestOperationVisibilityUsesFrozenPolicyAndSubjectSemanticsBeforePagination(t *testing.T) {
	const (
		namespaceID = "11111111-1111-4111-8111-111111111111"
		principalID = "22222222-2222-4222-8222-222222222222"
		operationID = "33333333-3333-4333-8333-333333333333"
		bindingID   = "44444444-4444-4444-8444-444444444444"
		policyID    = "55555555-5555-4555-8555-555555555555"
		userID      = "66666666-6666-4666-8666-666666666666"
	)
	visibility := policybulk.OperationVisibility{
		PrincipalID: principalID,
		Operation: accesscontrol.ResultScope{
			NamespaceID: namespaceID,
			UserIDs:     []accesscontrol.UserID{userID},
			ResourceIDs: map[accesscontrol.ScopeResourceType][]accesscontrol.ResourceID{
				accesscontrol.ScopeResourceOperation:           {operationID},
				accesscontrol.ScopeResourceAccessPolicyBinding: {bindingID},
				accesscontrol.ScopeResourceAccessPolicy:        {policyID},
			},
		},
		Access: accesscontrol.ResultScope{
			NamespaceID: namespaceID,
			ResourceIDs: map[accesscontrol.ScopeResourceType][]accesscontrol.ResourceID{
				accesscontrol.ScopeResourceAccessPolicy: {policyID},
			},
		},
		Rate: accesscontrol.ResultScope{NamespaceID: namespaceID},
	}
	statement := strings.Builder{}
	statement.WriteString("SELECT * FROM management_operations o WHERE o.namespace_id=$1")
	arguments := []any{namespaceID}
	appendOperationVisibility(&statement, &arguments, visibility)
	visibilityEnd := statement.Len()
	statement.WriteString(" ORDER BY o.created_at DESC,o.id ASC LIMIT $99")

	query := statement.String()
	if visibilityEnd >= strings.Index(query, "ORDER BY") || !strings.Contains(query, "NOT EXISTS") ||
		!strings.Contains(query, "o.origin_principal_id=") {
		t.Fatalf("operation visibility was not applied before pagination: %s", query)
	}
	bound := fmt.Sprint(arguments)
	if strings.Contains(bound, operationID) || strings.Contains(bound, bindingID) {
		t.Fatalf("exact operation or binding IDs widened frozen list semantics: %s", bound)
	}
	for _, expected := range []string{principalID, policyID, userID} {
		if !strings.Contains(bound, expected) {
			t.Fatalf("authorized policy/subject/origin %q was not bound: %s", expected, bound)
		}
	}
}
