package postgres

import (
	"context"
	"encoding/hex"
	"errors"
	"net/netip"
	"strings"
	"testing"
	"time"

	"github.com/DATA-DOG/go-sqlmock"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
)

func TestMutationMetaValidation(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*MutationMeta)
	}{
		{name: "missing reason", mutate: func(meta *MutationMeta) { meta.Reason = "" }},
		{name: "missing request id", mutate: func(meta *MutationMeta) { meta.RequestID = "" }},
		{name: "invalid action", mutate: func(meta *MutationMeta) { meta.Action = "User Create" }},
		{name: "mapped source IP", mutate: func(meta *MutationMeta) {
			meta.SourceIP = netip.MustParseAddr("::ffff:192.0.2.10")
		}},
		{name: "duplicate actor", mutate: func(meta *MutationMeta) {
			meta.ActorChain = append(meta.ActorChain, meta.ActorChain[0])
		}},
		{name: "sensitive detail", mutate: func(meta *MutationMeta) {
			meta.Details = AuditDetails{"secret": "must-not-be-recorded"}
		}},
	}
	if err := validateMutationMeta(testMutationMeta()); err != nil {
		t.Fatalf("valid metadata rejected: %v", err)
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			meta := testMutationMeta()
			test.mutate(&meta)
			if err := validateMutationMeta(meta); err == nil {
				t.Fatal("expected validation error")
			}
		})
	}
}

func TestAuditEventHashIsDeterministicAndTamperEvident(t *testing.T) {
	before := "4"
	document := auditHashDocument{
		EventID:          "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
		NamespaceID:      string(testNamespaceID),
		DesiredRevision:  "9",
		ChainSequence:    "3",
		ActorPrincipalID: string(testActorID),
		ActorChain:       []string{string(testActorID)},
		Action:           "user.update",
		ResourceType:     "user",
		ResourceID:       string(testUserID),
		RequestID:        "request-test",
		SourceIP:         "192.0.2.10",
		Outcome:          "allowed",
		BeforeRevision:   &before,
		AfterRevision:    "5",
		Details:          map[string]string{"field": "displayName"},
		PreviousHash:     strings.Repeat("ab", 32),
		CreatedAt:        testNow.Format("2006-01-02T15:04:05Z07:00"),
	}
	first, err := computeAuditEventHash(document)
	if err != nil {
		t.Fatal(err)
	}
	second, err := computeAuditEventHash(document)
	if err != nil {
		t.Fatal(err)
	}
	if first != second {
		t.Fatal("same canonical audit document produced different hashes")
	}
	document.Details["field"] = "email"
	tampered, err := computeAuditEventHash(document)
	if err != nil {
		t.Fatal(err)
	}
	if first == tampered {
		t.Fatal("tampered audit document retained its hash")
	}
}

func TestVerifyAuditChain(t *testing.T) {
	first := auditHashDocument{
		EventID:         "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
		NamespaceID:     string(testNamespaceID),
		DesiredRevision: "1",
		ChainSequence:   "1",
		Action:          "namespace.create",
		ResourceType:    "namespace",
		ResourceID:      string(testNamespaceID),
		RequestID:       "request-one",
		Outcome:         "allowed",
		AfterRevision:   "1",
		ActorChain:      []string{},
		Details:         map[string]string{},
		CreatedAt:       testNow.Format("2006-01-02T15:04:05Z07:00"),
	}
	firstHash, err := computeAuditEventHash(first)
	if err != nil {
		t.Fatal(err)
	}
	before := "1"
	second := auditHashDocument{
		EventID:         "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
		NamespaceID:     string(testNamespaceID),
		DesiredRevision: "2",
		ChainSequence:   "2",
		Action:          "namespace.update",
		ResourceType:    "namespace",
		ResourceID:      string(testNamespaceID),
		RequestID:       "request-two",
		Outcome:         "allowed",
		BeforeRevision:  &before,
		AfterRevision:   "2",
		ActorChain:      []string{},
		Details:         map[string]string{"field": "displayName"},
		PreviousHash:    hex.EncodeToString(firstHash[:]),
		CreatedAt:       testNow.Add(time.Second).Format("2006-01-02T15:04:05Z07:00"),
	}
	secondHash, err := computeAuditEventHash(second)
	if err != nil {
		t.Fatal(err)
	}
	entries := []auditChainEntry{
		{Document: first, Hash: firstHash[:]},
		{Document: second, Hash: secondHash[:]},
	}
	if err := verifyAuditChain(entries); err != nil {
		t.Fatalf("valid chain rejected: %v", err)
	}
	tampered := append([]auditChainEntry(nil), entries...)
	tampered[1].Document.RequestID = "tampered"
	if err := verifyAuditChain(tampered); err == nil {
		t.Fatal("tampered chain was accepted")
	}
	crossNamespace := append([]auditChainEntry(nil), entries...)
	crossNamespace[1].Document.NamespaceID = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
	if err := verifyAuditChain(crossNamespace); err == nil {
		t.Fatal("cross-namespace chain was accepted")
	}
}

func TestAuditInsertFailureRollsBackDesiredStateAndOutbox(t *testing.T) {
	store, mock := newMockStore(t)
	user := accesscontrol.User{
		NamespaceID: testNamespaceID, ID: testUserID, Email: "user@example.com",
		DisplayName: "User", Status: accesscontrol.UserStatusActive,
		CreatedAt: testNow, UpdatedAt: testNow,
	}
	mock.ExpectBegin()
	mock.ExpectExec(queryPattern(insertSubjectQuery)).
		WithArgs(testNamespaceID, testUserID, accesscontrol.SubjectKindUser, testNow).
		WillReturnResult(sqlmock.NewResult(0, 1))
	mock.ExpectQuery(queryPattern(insertUserQuery)).
		WillReturnRows(userRows().AddRow(
			user.ID, user.NamespaceID, user.Email, user.DisplayName,
			user.Status, int64(1), user.CreatedAt, user.UpdatedAt, nil,
		))
	mock.ExpectQuery(queryPattern(lockNamespaceQuery)).
		WillReturnRows(sqlmock.NewRows([]string{"runtime_epoch"}).AddRow(int64(7)))
	mock.ExpectQuery(queryPattern(nextRevisionQuery)).
		WillReturnRows(sqlmock.NewRows([]string{"revision"}).AddRow(int64(12)))
	mock.ExpectExec(queryPattern(insertRevisionQuery)).
		WillReturnResult(sqlmock.NewResult(0, 1))
	mock.ExpectExec(queryPattern(insertOutboxQuery)).
		WillReturnResult(sqlmock.NewResult(0, 1))
	mock.ExpectExec(queryPattern(insertAuditHeadQuery)).
		WillReturnResult(sqlmock.NewResult(0, 1))
	mock.ExpectQuery(queryPattern(lockAuditHeadQuery)).
		WillReturnRows(sqlmock.NewRows([]string{"event_count", "last_hash"}).AddRow(int64(0), nil))
	mock.ExpectQuery(queryPattern(auditTimestampQuery)).
		WillReturnRows(sqlmock.NewRows([]string{"clock_timestamp"}).AddRow(testNow))
	mock.ExpectExec(queryPattern(insertAuditEventQuery)).
		WillReturnError(errors.New("audit storage unavailable"))
	mock.ExpectRollback()

	_, err := store.CreateUser(context.Background(), user, testMutationMeta())
	if err == nil || !strings.Contains(err.Error(), "insert access audit event") {
		t.Fatalf("expected audit insertion failure, got %v", err)
	}
	assertExpectations(t, mock)
}
