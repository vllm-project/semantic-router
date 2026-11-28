package config

import (
	"strings"
	"testing"
)

// The global allow and block lists filtered every tool selection added. No
// v0.5 field filters what selection adds from the catalog, so migrate names
// them rather than dropping them in silence.
func TestMigrateNamesTheGlobalToolAllowAndBlockLists(t *testing.T) {
	doc := `version: v0.5
global:
  integrations:
    tools:
      enabled: true
      tools_db_path: config/tools_db.json
      advanced_filtering: {enabled: true, block_tools: [admin.delete]}
`
	_, notes, err := migrateConfigForms([]byte(doc), toolsForms)
	if err != nil {
		t.Fatalf("migrate: %v", err)
	}
	for _, note := range notes {
		if strings.Contains(note.Message, "block_tools") && strings.Contains(note.Message, "tools_db catalog") {
			return
		}
	}
	t.Fatalf("notes = %+v, want one naming block_tools and the catalog", notes)
}
