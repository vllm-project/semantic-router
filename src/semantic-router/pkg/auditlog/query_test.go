package auditlog

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/DATA-DOG/go-sqlmock"
)

func TestAuditQueryRequiresBoundedKeysetInput(t *testing.T) {
	query := Query{
		NamespaceID: "11111111-1111-4111-8111-111111111111",
		Start:       time.Date(2026, 8, 22, 0, 0, 0, 0, time.UTC),
		End:         time.Date(2026, 8, 23, 0, 0, 0, 0, time.UTC), PageSize: 50,
		Filters: Filters{Outcome: "allowed"},
	}
	if err := validateQuery(query); err != nil {
		t.Fatalf("valid query rejected: %v", err)
	}
	query.PageSize = 201
	if err := validateQuery(query); !errors.Is(err, ErrInvalidQuery) {
		t.Fatalf("oversized page error = %v", err)
	}
}

func TestAuditCursorBindsNamespaceAndFilters(t *testing.T) {
	codec, err := NewCursorCodec([]byte("0123456789abcdef0123456789abcdef"))
	if err != nil {
		t.Fatal(err)
	}
	start := time.Date(2026, 8, 22, 0, 0, 0, 0, time.UTC)
	end := start.Add(time.Hour)
	value := cursorValue{
		Version: 1, NamespaceID: "11111111-1111-4111-8111-111111111111",
		QueryDigest: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		Start:       start.UnixNano(), End: end.UnixNano(),
		CreatedAt: time.Now().UnixNano(), EventID: "22222222-2222-4222-8222-222222222222",
	}
	encoded, err := codec.encode(value)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := codec.decode(encoded)
	if err != nil || decoded != value {
		t.Fatalf("decode() = (%#v, %v)", decoded, err)
	}
	gotStart, gotEnd, err := codec.TimeRange(encoded)
	if err != nil || !gotStart.Equal(start) || !gotEnd.Equal(end) {
		t.Fatalf("TimeRange() = (%s, %s, %v), want (%s, %s)", gotStart, gotEnd, err, start, end)
	}
	tampered := []byte(encoded)
	tampered[len(tampered)-1] ^= 1
	if _, err := codec.decode(string(tampered)); !errors.Is(err, ErrInvalidQuery) {
		t.Fatalf("tampered cursor error = %v", err)
	}
}

func TestAuditListNormalizesPostgresInetToHostAddress(t *testing.T) {
	database, mock, err := sqlmock.New()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = database.Close() })
	start := time.Date(2026, 8, 26, 0, 0, 0, 0, time.UTC)
	createdAt := time.Date(2026, 8, 26, 12, 0, 0, 0, time.UTC)
	rows := sqlmock.NewRows([]string{
		"id", "namespace_id", "desired_revision", "chain_sequence",
		"actor_principal_id", "actor_chain", "action", "resource_type",
		"resource_id", "request_id", "source_ip", "outcome", "reason",
		"before_revision", "after_revision", "details", "previous_hash",
		"event_hash", "created_at",
	}).AddRow(
		"22222222-2222-4222-8222-222222222222",
		"11111111-1111-4111-8111-111111111111",
		nil,
		int64(1),
		"33333333-3333-4333-8333-333333333333",
		[]byte(`["33333333-3333-4333-8333-333333333333"]`),
		"api_key.create",
		"api_key",
		"44444444-4444-4444-8444-444444444444",
		"request-test",
		"192.0.2.10",
		"allowed",
		"Create API key.",
		nil,
		nil,
		[]byte(`{}`),
		nil,
		make([]byte, 32),
		createdAt,
	)
	mock.ExpectQuery(`(?s)SELECT .*COALESCE\(host\(source_ip\),''\).*FROM access_audit_events`).
		WithArgs("11111111-1111-4111-8111-111111111111", start, start.Add(24*time.Hour), 2).
		WillReturnRows(rows)
	codec, err := NewCursorCodec([]byte("0123456789abcdef0123456789abcdef"))
	if err != nil {
		t.Fatal(err)
	}
	page, err := (PostgresQueries{DB: database}).List(context.Background(), Query{
		NamespaceID: "11111111-1111-4111-8111-111111111111",
		Start:       start, End: start.Add(24 * time.Hour), PageSize: 1,
	}, codec)
	if err != nil {
		t.Fatalf("List() error = %v", err)
	}
	if len(page.Items) != 1 || page.Items[0].SourceIP != "192.0.2.10" {
		t.Fatalf("audit events = %#v", page.Items)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatal(err)
	}
}
