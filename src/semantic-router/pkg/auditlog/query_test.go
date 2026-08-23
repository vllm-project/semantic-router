package auditlog

import (
	"errors"
	"testing"
	"time"
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
	value := cursorValue{
		Version: 1, NamespaceID: "11111111-1111-4111-8111-111111111111",
		QueryDigest: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		CreatedAt:   time.Now().UnixNano(), EventID: "22222222-2222-4222-8222-222222222222",
	}
	encoded, err := codec.encode(value)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := codec.decode(encoded)
	if err != nil || decoded != value {
		t.Fatalf("decode() = (%#v, %v)", decoded, err)
	}
	tampered := []byte(encoded)
	tampered[len(tampered)-1] ^= 1
	if _, err := codec.decode(string(tampered)); !errors.Is(err, ErrInvalidQuery) {
		t.Fatalf("tampered cursor error = %v", err)
	}
}
