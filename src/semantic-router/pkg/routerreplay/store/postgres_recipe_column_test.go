package store

import (
	"database/sql"
	"strings"
	"testing"
	"time"
)

// TestPostgresRecordCarriesRecipe pins recipe identity across the PostgreSQL
// write and read paths.
//
// Background: Record.Recipe is populated by the ExtProc recorder on every
// request and the Router Replay API filters, searches, and builds its recipe
// dropdown from it. PostgreSQL was the only backend with an explicit column
// mapping (memory, Redis, Qdrant, and Milvus marshal the whole Record), so it
// was the only one that dropped the field: writes discarded it and reads always
// returned "", leaving ?recipe= matching nothing on Postgres-backed
// deployments.
func TestPostgresRecordCarriesRecipe(t *testing.T) {
	const recipe = "beta"

	idx := columnIndex(t, "recipe")

	built, err := newPostgresInsertRecord(Record{
		ID:        "rec-1",
		Timestamp: time.Now().UTC(),
		Recipe:    recipe,
	})
	if err != nil {
		t.Fatalf("newPostgresInsertRecord: %v", err)
	}

	args := built.args()
	if got := args[idx]; got != recipe {
		t.Errorf("insert arg for recipe column = %#v, want %q", got, recipe)
	}

	row := postgresRecordRow{}
	if got := row.scanDestinations()[idx]; got != &row.recipe {
		t.Errorf("scan destination for recipe column = %#v, want &row.recipe", got)
	}
}

// TestPostgresRecordRowDecodesRecipe covers the scan side: a row read back from
// PostgreSQL must land its recipe on the decoded Record.
func TestPostgresRecordRowDecodesRecipe(t *testing.T) {
	row := postgresRecordRow{
		signalsJSON: []byte("{}"),
		recipe:      sql.NullString{String: "beta", Valid: true},
	}

	record, err := row.decode()
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if record.Recipe != "beta" {
		t.Errorf("decoded Recipe = %q, want %q", record.Recipe, "beta")
	}
}

// TestPostgresRecordRowDecodesMissingRecipe covers rows written before the
// recipe column existed. The migration backfills them as NULL, which must decode
// to an empty recipe rather than failing the read.
func TestPostgresRecordRowDecodesMissingRecipe(t *testing.T) {
	row := postgresRecordRow{signalsJSON: []byte("{}")}

	record, err := row.decode()
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if record.Recipe != "" {
		t.Errorf("decoded Recipe = %q, want empty for a NULL column", record.Recipe)
	}
}

// TestPostgresCreateTableAddsRecipeColumn pins the schema upgrade. Existing
// deployments only gain the column through the ALTER block, so dropping it
// would leave every write failing against an already-created table.
func TestPostgresCreateTableAddsRecipeColumn(t *testing.T) {
	query := postgresCreateTableQuery(DefaultPostgresTableName)

	want := "ALTER TABLE " + DefaultPostgresTableName + " ADD COLUMN IF NOT EXISTS recipe TEXT;"
	if !strings.Contains(query, want) {
		t.Errorf("create-table query is missing the non-destructive recipe migration:\nwant substring: %s", want)
	}
}

// columnIndex resolves a column's position in the INSERT statement, which
// TestPostgresInsertSelectColumnAlignment pins to the SELECT list and the scan
// destinations.
func columnIndex(t *testing.T, name string) int {
	t.Helper()

	for i, col := range extractInsertColumns(t, postgresInsertQueryTemplate) {
		if col == name {
			return i
		}
	}
	t.Fatalf("column %q is not in the INSERT column list", name)
	return -1
}
