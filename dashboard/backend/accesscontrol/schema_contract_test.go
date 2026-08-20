package accesscontrol

import (
	"strings"
	"testing"
)

func TestSchemaCreatesCompleteAPIKeyContract(t *testing.T) {
	for _, fragment := range []string{
		"CREATE TABLE IF NOT EXISTS access_budgets",
		"secret_ciphertext TEXT NOT NULL",
		"budget_id TEXT REFERENCES access_budgets(id) ON DELETE SET NULL",
		"CREATE INDEX IF NOT EXISTS idx_access_keys_budget ON access_api_keys(budget_id)",
	} {
		if !strings.Contains(schema, fragment) {
			t.Fatalf("access-control schema is missing %q", fragment)
		}
	}
}
