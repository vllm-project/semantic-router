package store

import (
	"regexp"
	"strings"
	"testing"
	"time"
)

// TestPostgresInsertQueryColumnArgsAlignment pins the three-way invariant
// between the INSERT column list, the $N placeholder list, and the slice
// returned by postgresInsertRecord.args().
//
// Background: PR #1857 added projection_trace to the column list but forgot
// to add $51 to the VALUES clause. Postgres rejected every write with
// "INSERT has more target columns than expressions"; the error was swallowed
// by persistReplayRecord and the dashboard Insight panel silently stayed
// empty for 25 days before anyone noticed.
func TestPostgresInsertQueryColumnArgsAlignment(t *testing.T) {
	cols := extractInsertColumns(t, postgresInsertQueryTemplate)
	if len(cols) == 0 {
		t.Fatalf("could not extract column list from postgresInsertQueryTemplate")
	}

	placeholders := extractInsertPlaceholders(t, postgresInsertQueryTemplate)
	if len(placeholders) == 0 {
		t.Fatalf("could not extract placeholder list from postgresInsertQueryTemplate")
	}

	args := samplePostgresInsertArgs(t)

	if len(cols) != len(placeholders) {
		t.Errorf("column / placeholder count mismatch: %d columns vs %d placeholders\ncolumns=%v\nplaceholders=%v",
			len(cols), len(placeholders), cols, placeholders)
	}
	if len(cols) != len(args) {
		t.Errorf("column / args count mismatch: %d columns vs %d args from postgresInsertRecord.args()",
			len(cols), len(args))
	}
	if len(placeholders) != len(args) {
		t.Errorf("placeholder / args count mismatch: %d placeholders vs %d args", len(placeholders), len(args))
	}

	for i, p := range placeholders {
		want := "$" + itoa(i+1)
		if p != want {
			t.Errorf("placeholder #%d is %q, expected %q (placeholders must be sequential $1..$N)", i+1, p, want)
		}
	}
}

// extractInsertColumns parses the column list between
// `INSERT INTO %s (` and `) VALUES`.
func extractInsertColumns(t *testing.T, tmpl string) []string {
	t.Helper()
	start := strings.Index(tmpl, "INSERT INTO %s (")
	if start < 0 {
		return nil
	}
	start += len("INSERT INTO %s (")
	end := strings.Index(tmpl[start:], ") VALUES")
	if end < 0 {
		return nil
	}
	raw := tmpl[start : start+end]
	parts := strings.Split(raw, ",")
	cols := make([]string, 0, len(parts))
	for _, p := range parts {
		trimmed := strings.TrimSpace(p)
		if trimmed == "" {
			continue
		}
		cols = append(cols, trimmed)
	}
	return cols
}

// extractInsertPlaceholders parses the $N placeholders after `VALUES (`.
func extractInsertPlaceholders(t *testing.T, tmpl string) []string {
	t.Helper()
	re := regexp.MustCompile(`\$\d+`)
	valuesIdx := strings.Index(tmpl, "VALUES (")
	if valuesIdx < 0 {
		return nil
	}
	return re.FindAllString(tmpl[valuesIdx:], -1)
}

// samplePostgresInsertArgs builds a minimal Record, runs it through the
// production insert-record builder, and returns the resulting args slice.
// We do not assert specific values — only the slice length.
func samplePostgresInsertArgs(t *testing.T) []interface{} {
	t.Helper()
	rec := Record{
		ID:        "test-id",
		Timestamp: time.Now().UTC(),
	}
	built, err := newPostgresInsertRecord(rec)
	if err != nil {
		t.Fatalf("newPostgresInsertRecord: %v", err)
	}
	return built.args()
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := false
	if n < 0 {
		neg = true
		n = -n
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
}
