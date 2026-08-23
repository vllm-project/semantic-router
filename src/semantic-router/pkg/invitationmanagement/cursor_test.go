package invitationmanagement

import (
	"context"
	"encoding/base64"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/internal/testsupport/signedtoken"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

func TestInvitationCursorRejectsForgeryAndNonCanonicalEncoding(t *testing.T) {
	codec, testInvitationCursorRejectsForgeryAndNonCanonicalEncodingErr := newCursorCodec(invitationCursorKeyring("cursor-v1", "k"))
	if testInvitationCursorRejectsForgeryAndNonCanonicalEncodingErr != nil {
		t.Fatal(testInvitationCursorRejectsForgeryAndNonCanonicalEncodingErr)
	}
	t.Cleanup(codec.close)
	token, testInvitationCursorRejectsForgeryAndNonCanonicalEncodingErr := codec.encode(cursorPayload{
		Kind: "invitations", NamespaceID: "11111111-1111-4111-8111-111111111111",
		Status: StatusPending, ExpiresAt: time.Date(2026, 8, 24, 0, 0, 0, 0, time.UTC),
		ID: "22222222-2222-4222-8222-222222222222",
	})
	if testInvitationCursorRejectsForgeryAndNonCanonicalEncodingErr != nil {
		t.Fatal(testInvitationCursorRejectsForgeryAndNonCanonicalEncodingErr)
	}
	if _, err := codec.decode(token); err != nil {
		t.Fatal(err)
	}
	if _, err := codec.decode(signedtoken.Alias(t, token)); err == nil {
		t.Fatal("non-canonical invitation cursor signature was accepted")
	}
	parts := strings.Split(token, ".")
	payload, testInvitationCursorRejectsForgeryAndNonCanonicalEncodingErr := base64.RawURLEncoding.DecodeString(parts[2])
	if testInvitationCursorRejectsForgeryAndNonCanonicalEncodingErr != nil {
		t.Fatal(testInvitationCursorRejectsForgeryAndNonCanonicalEncodingErr)
	}
	parts[2] = base64.RawURLEncoding.EncodeToString(append(payload, ' '))
	if _, err := codec.decode(strings.Join(parts, ".")); err == nil {
		t.Fatal("forged invitation cursor payload was accepted")
	}
}

func TestInvitationCursorSupportsKeyRotation(t *testing.T) {
	oldCodec, testInvitationCursorSupportsKeyRotationErr := newCursorCodec(invitationCursorKeyring("cursor-v1", "a"))
	if testInvitationCursorSupportsKeyRotationErr != nil {
		t.Fatal(testInvitationCursorSupportsKeyRotationErr)
	}
	t.Cleanup(oldCodec.close)
	token, testInvitationCursorSupportsKeyRotationErr := oldCodec.encode(cursorPayload{Kind: "invitations"})
	if testInvitationCursorSupportsKeyRotationErr != nil {
		t.Fatal(testInvitationCursorSupportsKeyRotationErr)
	}
	rotated, testInvitationCursorSupportsKeyRotationErr := newCursorCodec(securitykeyring.Symmetric{
		ActiveVersion: "cursor-v2",
		Keys: map[string][]byte{
			"cursor-v1": []byte(strings.Repeat("a", 32)),
			"cursor-v2": []byte(strings.Repeat("b", 32)),
		},
	})
	if testInvitationCursorSupportsKeyRotationErr != nil {
		t.Fatal(testInvitationCursorSupportsKeyRotationErr)
	}
	t.Cleanup(rotated.close)
	if _, err := rotated.decode(token); err != nil {
		t.Fatalf("retained cursor key rejected during rotation: %v", err)
	}
	retired, testInvitationCursorSupportsKeyRotationErr := newCursorCodec(invitationCursorKeyring("cursor-v2", "b"))
	if testInvitationCursorSupportsKeyRotationErr != nil {
		t.Fatal(testInvitationCursorSupportsKeyRotationErr)
	}
	t.Cleanup(retired.close)
	if _, err := retired.decode(token); err == nil {
		t.Fatal("retired cursor key remained accepted")
	}
}

func TestInvitationListCursorBindsNamespaceAndStatus(t *testing.T) {
	repository := &invitationListRepository{page: RepositoryPage{
		Items: []Invitation{{
			ID:        "22222222-2222-4222-8222-222222222222",
			ExpiresAt: time.Date(2026, 8, 25, 0, 0, 0, 0, time.UTC),
		}},
		HasMore: true,
	}}
	service := newInvitationCursorTestService(t, repository)
	namespaceID := "11111111-1111-4111-8111-111111111111"
	page, err := service.List(context.Background(), ListRequest{
		NamespaceID: namespaceID, Status: StatusPending, PageSize: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	if page.NextCursor == "" {
		t.Fatal("first invitation page omitted its continuation cursor")
	}
	if _, err := service.List(context.Background(), ListRequest{
		NamespaceID: namespaceID, Status: StatusAccepted, PageSize: 1, Cursor: page.NextCursor,
	}); err == nil {
		t.Fatal("invitation cursor was reused with another status filter")
	}
	if _, err := service.List(context.Background(), ListRequest{
		NamespaceID: "33333333-3333-4333-8333-333333333333",
		Status:      StatusPending, PageSize: 1, Cursor: page.NextCursor,
	}); err == nil {
		t.Fatal("invitation cursor was reused in another Namespace")
	}
	if _, err := service.List(context.Background(), ListRequest{
		NamespaceID: namespaceID, Status: StatusPending, PageSize: 1, Cursor: page.NextCursor,
	}); err != nil {
		t.Fatal(err)
	}
	if repository.query.After == nil ||
		repository.query.After.ID != "22222222-2222-4222-8222-222222222222" {
		t.Fatalf("continuation query = %+v", repository.query)
	}
}

type invitationListRepository struct {
	lifecycleRepository
	page  RepositoryPage
	query InvitationQuery
}

func (repository *invitationListRepository) List(_ context.Context, query InvitationQuery) (RepositoryPage, error) {
	repository.query = query
	return repository.page, nil
}

func newInvitationCursorTestService(t *testing.T, repository Repository) *Service {
	t.Helper()
	commands, err := managementcommand.NewCodec(securitykeyring.Symmetric{
		ActiveVersion: "command-v1",
		Keys: map[string][]byte{
			"command-v1": []byte(strings.Repeat("c", 32)),
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = commands.Close() })
	service, err := NewService(Options{
		Repository: repository, Commands: commands,
		CursorKeyring: invitationCursorKeyring("cursor-v1", "k"),
		InvitationPeppers: accesscredential.PepperKeyring{
			ActiveVersion: "invite-v1",
			Keys: map[string][]byte{
				"invite-v1": []byte(strings.Repeat("i", 32)),
			},
		},
		ResponseKEK: accesscredential.KEKKeyring{
			ActiveVersion: "response-v1",
			Keys: map[string][]byte{
				"response-v1": []byte(strings.Repeat("r", 32)),
			},
		},
		IdempotencyTTL: time.Hour, SecretDeliveryTTL: time.Hour,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(service.Close)
	return service
}

func invitationCursorKeyring(version, fill string) securitykeyring.Symmetric {
	return securitykeyring.Symmetric{
		ActiveVersion: version,
		Keys: map[string][]byte{
			version: []byte(strings.Repeat(fill, 32)),
		},
	}
}
