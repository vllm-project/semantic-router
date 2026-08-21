package accesscontrol

import (
	"strings"
	"testing"
)

func TestSchemaCreatesCompleteAPIKeyContract(t *testing.T) {
	for _, fragment := range []string{
		"CREATE TABLE IF NOT EXISTS access_budgets",
		"scope_type IN ('global', 'user', 'team', 'key')",
		"subject_type IN ('user', 'team', 'key')",
		"CREATE UNIQUE INDEX IF NOT EXISTS idx_access_team_members_user ON access_team_members(user_id)",
		"secret_ciphertext TEXT NOT NULL",
		"budget_id TEXT REFERENCES access_budgets(id) ON DELETE SET NULL",
		"CREATE INDEX IF NOT EXISTS idx_access_keys_budget ON access_api_keys(budget_id)",
	} {
		if !strings.Contains(schema, fragment) {
			t.Fatalf("access-control schema is missing %q", fragment)
		}
	}
}
