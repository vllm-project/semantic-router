package postgres

import (
	"database/sql/driver"
	"encoding/json"
	"net/netip"
	"reflect"
	"regexp"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/DATA-DOG/go-sqlmock"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
)

const (
	testNamespaceID    = accesscontrol.NamespaceID("11111111-1111-4111-8111-111111111111")
	testUserID         = accesscontrol.UserID("22222222-2222-4222-8222-222222222222")
	testTeamID         = accesscontrol.TeamID("33333333-3333-4333-8333-333333333333")
	testAPIKeyID       = accesscontrol.APIKeyID("44444444-4444-4444-8444-444444444444")
	testCredentialID   = accesscontrol.CredentialVersionID("55555555-5555-4555-8555-555555555555")
	testAccessPolicyID = accesscontrol.AccessPolicyID("66666666-6666-4666-8666-666666666666")
	testResourceID     = accesscontrol.ResourceID("model/qwen-3.5")
	testRatePolicyID   = accesscontrol.RateLimitPolicyID("88888888-8888-4888-8888-888888888888")
	testRuleID         = accesscontrol.RateLimitRuleID("99999999-9999-4999-8999-999999999999")
	testBindingID      = accesscontrol.PolicyBindingID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
	testActorID        = accesscontrol.ManagementPrincipalID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
)

var testNow = time.Date(2026, time.August, 22, 10, 0, 0, 0, time.UTC)

func newMockStore(t *testing.T) (*Store, sqlmock.Sqlmock) {
	t.Helper()
	db, mock, err := sqlmock.New(sqlmock.QueryMatcherOption(sqlmock.QueryMatcherRegexp))
	if err != nil {
		t.Fatalf("create SQL mock: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })
	store, err := New(db)
	if err != nil {
		t.Fatalf("create store: %v", err)
	}
	return store, mock
}

func queryPattern(query string) string { return regexp.QuoteMeta(query) }

func testMutationMeta() MutationMeta {
	actor := testActorID
	return MutationMeta{
		ActorPrincipalID: &actor,
		ActorChain:       []accesscontrol.ManagementPrincipalID{actor},
		RequestID:        "request-test",
		SourceIP:         netip.MustParseAddr("192.0.2.10"),
		Action:           "test.mutate",
		Reason:           "test mutation",
		Details:          AuditDetails{"test": "true"},
	}
}

func expectOutbox(
	mock sqlmock.Sqlmock,
	aggregateType string,
	aggregateID string,
	aggregateRevision int64,
	desiredRevision int64,
	operation outboxOperation,
	references map[string]string,
) {
	mock.ExpectQuery(queryPattern(lockNamespaceQuery)).
		WithArgs(testNamespaceID).
		WillReturnRows(sqlmock.NewRows([]string{"runtime_epoch"}).AddRow(int64(7)))
	mock.ExpectQuery(queryPattern(nextRevisionQuery)).
		WithArgs(testNamespaceID).
		WillReturnRows(sqlmock.NewRows([]string{"revision"}).AddRow(desiredRevision))
	mock.ExpectExec(queryPattern(insertRevisionQuery)).
		WithArgs(testNamespaceID, desiredRevision, int64(7), "test mutation", testActorID).
		WillReturnResult(sqlmock.NewResult(0, 1))
	mock.ExpectExec(queryPattern(insertOutboxQuery)).
		WithArgs(
			sqlmock.AnyArg(), testNamespaceID, desiredRevision, aggregateType,
			aggregateID, operation, outboxPayloadMatcher{
				revision:   aggregateRevision,
				references: references,
			},
		).
		WillReturnResult(sqlmock.NewResult(0, 1))
	expectAudit(mock, aggregateType, aggregateID, aggregateRevision, desiredRevision, operation)
}

func expectAudit(
	mock sqlmock.Sqlmock,
	aggregateType string,
	aggregateID string,
	aggregateRevision int64,
	desiredRevision int64,
	operation outboxOperation,
) {
	mock.ExpectExec(queryPattern(insertAuditHeadQuery)).
		WithArgs(testNamespaceID).
		WillReturnResult(sqlmock.NewResult(0, 1))
	mock.ExpectQuery(queryPattern(lockAuditHeadQuery)).
		WithArgs(testNamespaceID).
		WillReturnRows(sqlmock.NewRows([]string{"event_count", "last_hash"}).AddRow(int64(0), nil))
	mock.ExpectQuery(queryPattern(auditTimestampQuery)).
		WillReturnRows(sqlmock.NewRows([]string{"clock_timestamp"}).AddRow(testNow))
	beforeRevision := any(nil)
	if operation != outboxCreated {
		beforeRevision = aggregateRevision - 1
	}
	mock.ExpectExec(queryPattern(insertAuditEventQuery)).
		WithArgs(
			sqlmock.AnyArg(), testNamespaceID, desiredRevision, int64(1),
			testActorID, `["`+string(testActorID)+`"]`, "test.mutate",
			aggregateType, aggregateID, "request-test", "192.0.2.10", "allowed",
			"test mutation", beforeRevision, aggregateRevision, `{"test":"true"}`, nil,
			sha256ValueMatcher{}, testNow,
		).
		WillReturnResult(sqlmock.NewResult(0, 1))
	mock.ExpectExec(queryPattern(updateAuditHeadQuery)).
		WithArgs(testNamespaceID, sqlmock.AnyArg(), sha256ValueMatcher{}, testNow, int64(0)).
		WillReturnResult(sqlmock.NewResult(0, 1))
}

type outboxPayloadMatcher struct {
	revision   int64
	references map[string]string
}

type sha256ValueMatcher struct{}

func (sha256ValueMatcher) Match(value driver.Value) bool {
	digest, ok := value.([]byte)
	return ok && len(digest) == 32
}

func (matcher outboxPayloadMatcher) Match(value driver.Value) bool {
	encoded, ok := value.([]byte)
	if !ok {
		return false
	}
	text := string(encoded)
	for _, forbidden := range []string{"secret", "hmac", "ciphertext", "nonce", "pepper", "kek"} {
		if strings.Contains(strings.ToLower(text), forbidden) {
			return false
		}
	}
	var payload outboxPayload
	if err := json.Unmarshal(encoded, &payload); err != nil {
		return false
	}
	return payload.AggregateRevision == strconv.FormatInt(matcher.revision, 10) &&
		reflect.DeepEqual(payload.References, matcher.references)
}

func assertExpectations(t *testing.T, mock sqlmock.Sqlmock) {
	t.Helper()
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet SQL expectations: %v", err)
	}
}
