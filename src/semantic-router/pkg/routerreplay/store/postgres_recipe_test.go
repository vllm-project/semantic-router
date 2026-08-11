package store

import (
	"database/sql"
	"fmt"
	"strings"
	"testing"
	"time"
)

func TestPostgresRecipeColumnAlignmentAndMigration(t *testing.T) {
	const (
		addRecipeColumn   = "ALTER TABLE {{table}} ADD COLUMN IF NOT EXISTS recipe TEXT;"
		widenRecipeColumn = "ALTER TABLE {{table}} ALTER COLUMN recipe TYPE TEXT;"
	)
	if !strings.Contains(postgresCreateTableQueryTemplate, "recipe TEXT,") {
		t.Fatal("CREATE TABLE template does not provision the recipe column as TEXT")
	}
	addIndex := strings.Index(postgresCreateTableQueryTemplate, addRecipeColumn)
	if addIndex < 0 {
		t.Fatal("CREATE TABLE template does not migrate existing replay tables with the recipe column")
	}
	widenIndex := strings.Index(postgresCreateTableQueryTemplate, widenRecipeColumn)
	if widenIndex < 0 {
		t.Fatal("CREATE TABLE template does not widen existing recipe columns to TEXT")
	}
	if addIndex > widenIndex {
		t.Fatal("recipe column migration widens the column before ensuring it exists")
	}

	insertColumns := extractInsertColumns(t, postgresInsertQueryTemplate)
	selectColumns := splitSQLColumnList(postgresRecordSelectColumns)
	placeholders := extractInsertPlaceholders(t, postgresInsertQueryTemplate)
	longRecipe := strings.Repeat("named-recipe/", 32)
	if len(longRecipe) <= 255 {
		t.Fatal("long recipe fixture must exceed the legacy VARCHAR limit")
	}
	insertArgs := postgresRecipeInsertArgs(t, longRecipe)

	if len(insertColumns) != len(placeholders) || len(insertColumns) != len(insertArgs) {
		t.Fatalf(
			"recipe INSERT alignment mismatch: columns=%d placeholders=%d args=%d",
			len(insertColumns),
			len(placeholders),
			len(insertArgs),
		)
	}
	if len(selectColumns) != len((&postgresRecordRow{}).scanDestinations()) {
		t.Fatalf(
			"recipe SELECT alignment mismatch: columns=%d scan destinations=%d",
			len(selectColumns),
			len((&postgresRecordRow{}).scanDestinations()),
		)
	}

	insertRecipeIndex := sqlColumnIndex(t, insertColumns, "recipe")
	if got := insertArgs[insertRecipeIndex]; got != longRecipe {
		t.Fatalf("recipe INSERT argument did not preserve the full TEXT value")
	}
	selectRecipeIndex := sqlColumnIndex(t, selectColumns, "recipe")
	if _, ok := (&postgresRecordRow{}).scanDestinations()[selectRecipeIndex].(*sql.NullString); !ok {
		t.Fatalf("recipe SELECT destination has type %T, want *sql.NullString", (&postgresRecordRow{}).scanDestinations()[selectRecipeIndex])
	}
}

func TestPostgresRecipeRoundTripAndLegacyNull(t *testing.T) {
	insertColumns := extractInsertColumns(t, postgresInsertQueryTemplate)
	selectColumns := splitSQLColumnList(postgresRecordSelectColumns)
	insertRecipeIndex := sqlColumnIndex(t, insertColumns, "recipe")
	selectRecipeIndex := sqlColumnIndex(t, selectColumns, "recipe")
	longRecipe := strings.Repeat("recipe-segment/", 32)

	tests := []struct {
		name        string
		storedValue interface{}
		wantRecipe  string
	}{
		{
			name:        "long named recipe",
			storedValue: postgresRecipeInsertArgs(t, longRecipe)[insertRecipeIndex],
			wantRecipe:  longRecipe,
		},
		{
			name:        "legacy row without recipe",
			storedValue: nil,
			wantRecipe:  "",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			record, err := scanPostgresRecord(postgresRecipeScanner{
				recipeIndex: selectRecipeIndex,
				recipeValue: test.storedValue,
			})
			if err != nil {
				t.Fatalf("scanPostgresRecord: %v", err)
			}
			if record.Recipe != test.wantRecipe {
				t.Fatalf("round-tripped recipe = %q, want %q", record.Recipe, test.wantRecipe)
			}
		})
	}
}

func postgresRecipeInsertArgs(t *testing.T, recipe string) []interface{} {
	t.Helper()
	prepared, err := newPostgresInsertRecord(Record{
		ID:        "recipe-round-trip",
		Timestamp: time.Unix(1, 0).UTC(),
		Recipe:    recipe,
	})
	if err != nil {
		t.Fatalf("newPostgresInsertRecord: %v", err)
	}
	return prepared.args()
}

func sqlColumnIndex(t *testing.T, columns []string, want string) int {
	t.Helper()
	for index, column := range columns {
		if column == want {
			return index
		}
	}
	t.Fatalf("column %q not found in %v", want, columns)
	return -1
}

type postgresRecipeScanner struct {
	recipeIndex int
	recipeValue interface{}
}

func (scanner postgresRecipeScanner) Scan(destinations ...interface{}) error {
	for _, destination := range destinations {
		if jsonDestination, ok := destination.(*[]byte); ok {
			*jsonDestination = []byte("null")
		}
	}

	recipeDestination, ok := destinations[scanner.recipeIndex].(*sql.NullString)
	if !ok {
		return fmt.Errorf(
			"recipe scan destination has type %T, want *sql.NullString",
			destinations[scanner.recipeIndex],
		)
	}
	if scanner.recipeValue == nil {
		*recipeDestination = sql.NullString{}
		return nil
	}
	recipe, ok := scanner.recipeValue.(string)
	if !ok {
		return fmt.Errorf("recipe scan value has type %T, want string", scanner.recipeValue)
	}
	*recipeDestination = sql.NullString{String: recipe, Valid: true}
	return nil
}
