package postgres

import (
	"context"
	"encoding/hex"
	"encoding/json"
	"regexp"
	"slices"
	"strings"
	"testing"

	"github.com/DATA-DOG/go-sqlmock"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
)

const testToolSourceID = "ee18f660-d573-4751-8aed-15e8a57f22b8"

func TestSourceRevisionDiscoveryInvalidatesApprovalOnConnectionChange(t *testing.T) {
	current, input := testedToolSource()
	tests := []struct {
		name   string
		mutate func(*agentmanagement.ToolSourceInput)
	}{
		{name: "transport", mutate: func(value *agentmanagement.ToolSourceInput) { value.Transport = "other" }},
		{name: "endpoint", mutate: func(value *agentmanagement.ToolSourceInput) { value.Endpoint = "https://next.example.com/mcp" }},
		{name: "credential", mutate: func(value *agentmanagement.ToolSourceInput) {
			value.CredentialID = "8b94b44a-27a0-4652-b26c-97838e76fd30"
		}},
		{name: "allowed hosts", mutate: func(value *agentmanagement.ToolSourceInput) {
			value.EgressPolicy.AllowedHosts = []string{"next.example.com"}
		}},
		{name: "allowed ports", mutate: func(value *agentmanagement.ToolSourceInput) { value.EgressPolicy.AllowedPorts = []int{8443} }},
		{name: "private CIDRs", mutate: func(value *agentmanagement.ToolSourceInput) {
			value.EgressPolicy.AllowedPrivateCIDRs = []string{"10.24.0.0/16"}
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			changed := cloneToolSourceInput(input)
			test.mutate(&changed)
			tools, digest, invalidate, err := sourceRevisionDiscovery(current, changed)
			if err != nil {
				t.Fatalf("sourceRevisionDiscovery() error = %v", err)
			}
			if !invalidate || len(tools) != 0 || len(digest) != 0 {
				t.Fatalf("connection change retained discovery: tools=%v digest=%x invalidate=%t", tools, digest, invalidate)
			}
		})
	}
}

func TestSourceRevisionDiscoveryPreservesApprovalForMetadataOnlyChange(t *testing.T) {
	current, input := testedToolSource()
	input.Name = "Renamed source"
	input.Description = "Updated description"
	tools, digest, invalidate, err := sourceRevisionDiscovery(current, input)
	if err != nil {
		t.Fatalf("sourceRevisionDiscovery() error = %v", err)
	}
	if invalidate || !slices.EqualFunc(tools, current.DiscoveredTools, func(left, right agentmanagement.ToolDefinition) bool {
		return left.Name == right.Name
	}) {
		t.Fatalf("metadata-only change invalidated discovery: tools=%v invalidate=%t", tools, invalidate)
	}
	wantDigest, err := hex.DecodeString(strings.Repeat("ab", 32))
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(digest, wantDigest) {
		t.Fatalf("retained discovery digest = %x, want %x", digest, wantDigest)
	}

	tools[0].Name = "remote.changed"
	if current.DiscoveredTools[0].Name == tools[0].Name {
		t.Fatal("retained discovery returned shared mutable slice storage")
	}
}

func TestSourceRevisionDiscoveryFailsClosedOnCorruptRetainedDigest(t *testing.T) {
	current, input := testedToolSource()
	current.DiscoveryDigest = "corrupt"
	if _, _, _, err := sourceRevisionDiscovery(current, input); err == nil {
		t.Fatal("sourceRevisionDiscovery() retained a malformed discovery digest")
	}

	input.Endpoint = "https://next.example.com/mcp"
	tools, digest, invalidate, err := sourceRevisionDiscovery(current, input)
	if err != nil || !invalidate || len(tools) != 0 || len(digest) != 0 {
		t.Fatalf("connection reset with corrupt old digest = (%v, %x, %t, %v)", tools, digest, invalidate, err)
	}
}

func TestEncodeSourceRevisionExcludesMutablePresentationMetadata(t *testing.T) {
	_, input := testedToolSource()
	tools := []agentmanagement.ToolDefinition{{Name: "remote.source.read"}}
	_, original, err := encodeSourceRevision(input, tools)
	if err != nil {
		t.Fatalf("encodeSourceRevision() error = %v", err)
	}
	input.Name = "Renamed source"
	input.Description = "New presentation text"
	_, renamed, err := encodeSourceRevision(input, tools)
	if err != nil {
		t.Fatalf("encodeSourceRevision(renamed) error = %v", err)
	}
	if original != renamed {
		t.Fatal("mutable Tool Source presentation metadata changed immutable content digest")
	}
	input.Endpoint = "https://next.example.com/mcp"
	_, changed, err := encodeSourceRevision(input, tools)
	if err != nil {
		t.Fatalf("encodeSourceRevision(changed) error = %v", err)
	}
	if original == changed {
		t.Fatal("Tool Source connection change retained immutable content digest")
	}
}

func TestEncodeSourceRevisionCanonicalizesEmptyDiscovery(t *testing.T) {
	_, input := testedToolSource()
	_, nilTools, err := encodeSourceRevision(input, nil)
	if err != nil {
		t.Fatalf("encodeSourceRevision(nil) error = %v", err)
	}
	_, emptyTools, err := encodeSourceRevision(input, []agentmanagement.ToolDefinition{})
	if err != nil {
		t.Fatalf("encodeSourceRevision(empty) error = %v", err)
	}
	if nilTools != emptyTools {
		t.Fatal("nil and empty Tool Source discoveries produced different content digests")
	}
}

func TestEnsureToolSourceRevisionReusesExactHistoricalContent(t *testing.T) {
	database, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock.New() error = %v", err)
	}
	mock.ExpectBegin()
	tx, err := database.Begin()
	if err != nil {
		t.Fatalf("Begin() error = %v", err)
	}
	_, input := testedToolSource()
	tools := []agentmanagement.ToolDefinition{{Name: "remote.source.read"}}
	_, digest, err := encodeSourceRevision(input, tools)
	if err != nil {
		t.Fatalf("encodeSourceRevision() error = %v", err)
	}
	mock.ExpectQuery(regexp.QuoteMeta("SELECT revision FROM agent_tool_source_revisions")).
		WithArgs(testNamespaceID, testToolSourceID, digest[:]).
		WillReturnRows(sqlmock.NewRows([]string{"revision"}).AddRow(int64(2)))
	revision, err := ensureToolSourceRevision(
		context.Background(), tx, testNamespaceID, testToolSourceID, input, tools, nil,
		agentmanagement.MutationContext{},
	)
	if err != nil || revision != 2 {
		t.Fatalf("ensureToolSourceRevision() = (%d, %v), want (2, nil)", revision, err)
	}
	mock.ExpectRollback()
	if err := tx.Rollback(); err != nil {
		t.Fatalf("Rollback() error = %v", err)
	}
	mock.ExpectClose()
	if err := database.Close(); err != nil {
		t.Fatalf("close SQL mock: %v", err)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("historical revision was not reused: %v", err)
	}
}

func TestEnsureToolSourceRevisionAllocatesAfterHistoricalMaximum(t *testing.T) {
	database, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock.New() error = %v", err)
	}
	mock.ExpectBegin()
	tx, err := database.Begin()
	if err != nil {
		t.Fatalf("Begin() error = %v", err)
	}
	_, input := testedToolSource()
	tools := []agentmanagement.ToolDefinition{{Name: "remote.source.write"}}
	encoded, digest, err := encodeSourceRevision(input, tools)
	if err != nil {
		t.Fatalf("encodeSourceRevision() error = %v", err)
	}
	toolBytes, err := json.Marshal(tools)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	discoveryDigest := strings.Repeat("d", 32)
	mock.ExpectQuery(regexp.QuoteMeta("SELECT revision FROM agent_tool_source_revisions")).
		WithArgs(testNamespaceID, testToolSourceID, digest[:]).
		WillReturnRows(sqlmock.NewRows([]string{"revision"}))
	mock.ExpectQuery(regexp.QuoteMeta("SELECT COALESCE(max(revision),0)+1")).
		WithArgs(testNamespaceID, testToolSourceID).
		WillReturnRows(sqlmock.NewRows([]string{"revision"}).AddRow(int64(7)))
	mock.ExpectExec(regexp.QuoteMeta("INSERT INTO agent_tool_source_revisions")).
		WithArgs(
			testToolSourceID, testNamespaceID, int64(7), input.Transport, input.Endpoint,
			input.CredentialID, encoded.egress, toolBytes, []byte(discoveryDigest), digest[:], nil,
		).
		WillReturnResult(sqlmock.NewResult(0, 1))
	revision, err := ensureToolSourceRevision(
		context.Background(), tx, testNamespaceID, testToolSourceID, input, tools,
		[]byte(discoveryDigest), agentmanagement.MutationContext{},
	)
	if err != nil || revision != 7 {
		t.Fatalf("ensureToolSourceRevision() = (%d, %v), want (7, nil)", revision, err)
	}
	mock.ExpectRollback()
	if err := tx.Rollback(); err != nil {
		t.Fatalf("Rollback() error = %v", err)
	}
	mock.ExpectClose()
	if err := database.Close(); err != nil {
		t.Fatalf("close SQL mock: %v", err)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("new revision did not allocate after historical maximum: %v", err)
	}
}

func testedToolSource() (agentmanagement.ToolSource, agentmanagement.ToolSourceInput) {
	egress := agentmanagement.EgressPolicy{
		AllowedHosts:        []string{"tools.example.com"},
		AllowedPorts:        []int{443},
		AllowedPrivateCIDRs: []string{},
	}
	current := agentmanagement.ToolSource{
		ResourceIdentity: agentmanagement.ResourceIdentity{
			Name: "Tools", Description: "Tested source", Status: agentmanagement.StatusActive,
		},
		Kind: "remote", Transport: "streamable_http", Endpoint: "https://tools.example.com/mcp",
		CredentialID: "72cc3315-4332-4aa6-9e2b-f945948202a5", EgressPolicy: egress,
		DiscoveredTools:         []agentmanagement.ToolDefinition{{Name: "remote.source.read"}},
		DiscoveryDigest:         "sha256:" + strings.Repeat("ab", 32),
		ApprovedDiscoveryDigest: "sha256:" + strings.Repeat("ab", 32),
	}
	return current, agentmanagement.ToolSourceInput{
		Name: current.Name, Description: current.Description, Kind: current.Kind,
		Transport: current.Transport, Endpoint: current.Endpoint, CredentialID: current.CredentialID,
		EgressPolicy: cloneEgressPolicy(egress),
	}
}

func cloneToolSourceInput(source agentmanagement.ToolSourceInput) agentmanagement.ToolSourceInput {
	result := source
	result.EgressPolicy = cloneEgressPolicy(source.EgressPolicy)
	return result
}

func cloneEgressPolicy(source agentmanagement.EgressPolicy) agentmanagement.EgressPolicy {
	return agentmanagement.EgressPolicy{
		AllowedHosts:        append([]string(nil), source.AllowedHosts...),
		AllowedPorts:        append([]int(nil), source.AllowedPorts...),
		AllowedPrivateCIDRs: append([]string(nil), source.AllowedPrivateCIDRs...),
	}
}
