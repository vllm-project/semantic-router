package postgres

import (
	"context"
	"errors"
	"reflect"
	"regexp"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/DATA-DOG/go-sqlmock"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
)

const testNamespaceID = "8ba5622d-7619-4518-a0c8-7300c9bfe815"

func TestBuiltinBuilderSkillUsesExactRouterToolAllowlist(t *testing.T) {
	skill, err := builtinBuilderSkillInput()
	if err != nil {
		t.Fatalf("builtinBuilderSkillInput() error = %v", err)
	}
	want := []string{
		"router.catalog.describe",
		"router.entrypoint.prepare",
		"router.models.list",
		"router.publish.prepare",
		"router.recipe.evaluate",
		"router.recipe.get",
		"router.recipe.prepare",
		"router.recipe.probe",
		"router.recipe.validate",
		"router.recipes.examples",
		"router.skills.read",
	}
	if !slices.Equal(skill.RequiredTools, want) {
		t.Fatalf("built-in Builder tools = %v, want %v", skill.RequiredTools, want)
	}
	for _, tool := range skill.RequiredTools {
		if strings.ContainsAny(tool, "*?") {
			t.Fatalf("built-in Builder tool %q is not an exact name", tool)
		}
	}

	first := builtinBuilderToolNames()
	first[0] = "router.future.mutating-tool"
	if builtinBuilderToolNames()[0] == first[0] {
		t.Fatal("builtinBuilderToolNames returned shared mutable state")
	}
}

func TestDefaultProfileDefinitionsAreDeterministicAndLeastPrivilege(t *testing.T) {
	first, err := defaultProfileDefinitions(testNamespaceID)
	if err != nil {
		t.Fatalf("defaultProfileDefinitions() error = %v", err)
	}
	second, err := defaultProfileDefinitions(testNamespaceID)
	if err != nil {
		t.Fatalf("second defaultProfileDefinitions() error = %v", err)
	}
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("default Profile definitions are not deterministic:\nfirst: %#v\nsecond: %#v", first, second)
	}
	if len(first) != 2 || first[0].id == first[1].id {
		t.Fatalf("default Profile identities = %#v, want two distinct resources", first)
	}

	skill, err := builtinBuilderSkillInput()
	if err != nil {
		t.Fatalf("builtinBuilderSkillInput() error = %v", err)
	}
	profiles := make(map[agentmanagement.SessionMode]agentmanagement.ProfileInput, len(first))
	for _, definition := range first {
		if definition.input.DefaultTarget != nil {
			t.Fatalf("default Profile %q pins a target", definition.input.Name)
		}
		if len(definition.input.DefaultForModes) != 1 {
			t.Fatalf("default Profile %q owns modes %v", definition.input.Name, definition.input.DefaultForModes)
		}
		mode := definition.input.DefaultForModes[0]
		if _, duplicate := profiles[mode]; duplicate {
			t.Fatalf("multiple default Profiles own mode %q", mode)
		}
		profiles[mode] = definition.input
	}

	chat := profiles[agentmanagement.SessionChat]
	if !slices.Equal(chat.ToolPolicy.Allow, []string{"router.skills.read"}) || len(chat.Skills) != 0 {
		t.Fatalf("Chat Profile authority = tools %v, Skills %v", chat.ToolPolicy.Allow, chat.Skills)
	}
	builder := profiles[agentmanagement.SessionBuilder]
	if !slices.Equal(builder.ToolPolicy.Allow, skill.RequiredTools) {
		t.Fatalf("Builder Profile tools = %v, built-in Skill requires %v", builder.ToolPolicy.Allow, skill.RequiredTools)
	}
	wantSkills := []agentmanagement.SkillReference{{ID: builtinBuilderSkillID, Revision: 1}}
	if !reflect.DeepEqual(builder.Skills, wantSkills) {
		t.Fatalf("Builder Profile Skills = %#v, want %#v", builder.Skills, wantSkills)
	}
	if len(builder.ToolPolicy.Deny) != 0 {
		t.Fatalf("Builder Profile deny policy = %v, want empty with exact allowlist", builder.ToolPolicy.Deny)
	}

	for index := range first {
		_, firstDigest, err := encodeProfileRevision(first[index].input)
		if err != nil {
			t.Fatalf("encode Profile %q: %v", first[index].input.Name, err)
		}
		_, secondDigest, err := encodeProfileRevision(second[index].input)
		if err != nil {
			t.Fatalf("encode second Profile %q: %v", second[index].input.Name, err)
		}
		if firstDigest != secondDigest {
			t.Fatalf("default Profile %q digest is not deterministic", first[index].input.Name)
		}
	}
}

func TestDefaultProfileDefinitionsRejectInvalidNamespace(t *testing.T) {
	if _, err := defaultProfileDefinitions("not-a-uuid"); err == nil {
		t.Fatal("defaultProfileDefinitions() accepted an invalid Namespace")
	}
}

func TestDefaultReconcilerReadyFailsClosedWhenUnavailable(t *testing.T) {
	var reconciler *DefaultReconciler
	if err := reconciler.Ready(context.Background()); err == nil {
		t.Fatal("nil DefaultReconciler reported ready")
	}
	if err := (&DefaultReconciler{}).Ready(context.Background()); err == nil {
		t.Fatal("DefaultReconciler without a store reported ready")
	}
}

func TestDefaultReconcilerAcceptsDeterministicResourcesAcrossReplicas(t *testing.T) {
	database, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock.New() error = %v", err)
	}
	fixedNow := time.Date(2026, time.August, 23, 12, 0, 0, 0, time.UTC)
	store := &Store{db: database}
	first, err := NewDefaultReconciler(store, time.Minute, func() time.Time { return fixedNow })
	if err != nil {
		t.Fatalf("NewDefaultReconciler() error = %v", err)
	}
	second, err := NewDefaultReconciler(store, time.Minute, func() time.Time { return fixedNow })
	if err != nil {
		t.Fatalf("second NewDefaultReconciler() error = %v", err)
	}

	for _, reconciler := range []*DefaultReconciler{first, second} {
		expectExistingBuiltinSkill(t, mock)
		if err := reconciler.ensureBuiltinSkill(context.Background()); err != nil {
			t.Fatalf("ensureBuiltinSkill() error = %v", err)
		}
		expectExistingNamespaceProfiles(t, mock, fixedNow)
		if err := reconciler.ensureNamespaceDefaults(context.Background(), testNamespaceID); err != nil {
			t.Fatalf("ensureNamespaceDefaults() error = %v", err)
		}
	}
	mock.ExpectClose()
	if err := database.Close(); err != nil {
		t.Fatalf("close SQL mock: %v", err)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("default reconciliation SQL did not remain idempotent: %v", err)
	}
}

func expectExistingBuiltinSkill(t *testing.T, mock sqlmock.Sqlmock) {
	t.Helper()
	input, err := builtinBuilderSkillInput()
	if err != nil {
		t.Fatalf("builtinBuilderSkillInput() error = %v", err)
	}
	required, minimum, digest, err := encodeSkillRevision(input)
	if err != nil {
		t.Fatalf("encodeSkillRevision() error = %v", err)
	}
	mock.ExpectBegin()
	mock.ExpectExec(regexp.QuoteMeta("INSERT INTO agent_skills")).
		WithArgs(builtinBuilderSkillID, input.Name, input.Description).
		WillReturnResult(sqlmock.NewResult(0, 0))
	mock.ExpectExec(regexp.QuoteMeta("INSERT INTO agent_skill_revisions")).
		WithArgs(builtinBuilderSkillID, input.Instructions, required, minimum, digest[:]).
		WillReturnResult(sqlmock.NewResult(0, 0))
	mock.ExpectQuery(regexp.QuoteMeta("SELECT skill.name,skill.description,skill.builtin")).
		WithArgs(builtinBuilderSkillID).
		WillReturnRows(sqlmock.NewRows([]string{
			"name", "description", "builtin", "status", "current_revision", "content_digest",
		}).AddRow(input.Name, input.Description, true, string(agentmanagement.StatusActive), int64(1), digest[:]))
	mock.ExpectCommit()
}

func expectExistingNamespaceProfiles(t *testing.T, mock sqlmock.Sqlmock, now time.Time) {
	t.Helper()
	definitions, err := defaultProfileDefinitions(testNamespaceID)
	if err != nil {
		t.Fatalf("defaultProfileDefinitions() error = %v", err)
	}
	mock.ExpectBegin()
	for _, definition := range definitions {
		encoded, digest, err := encodeProfileRevision(definition.input)
		if err != nil {
			t.Fatalf("encodeProfileRevision(%q) error = %v", definition.input.Name, err)
		}
		mock.ExpectExec(regexp.QuoteMeta("INSERT INTO agent_profiles")).
			WithArgs(definition.id, testNamespaceID, definition.input.Name, definition.input.Description).
			WillReturnResult(sqlmock.NewResult(0, 0))
		mock.ExpectExec(regexp.QuoteMeta("INSERT INTO agent_profile_revisions")).
			WithArgs(
				definition.id, testNamespaceID, encoded.minimumCapabilities, encoded.supportedModes,
				encoded.toolPolicy, definition.input.ApprovalPolicy, definition.input.MaximumTurnSeconds,
				definition.input.MaximumToolSteps, definition.input.ContextTokenBudget, digest[:],
			).
			WillReturnResult(sqlmock.NewResult(0, 0))
		for ordinal, skill := range definition.input.Skills {
			mock.ExpectExec(regexp.QuoteMeta("INSERT INTO agent_profile_skills")).
				WithArgs(testNamespaceID, definition.id, ordinal, skill.ID, skill.Revision).
				WillReturnResult(sqlmock.NewResult(0, 0))
		}
		mock.ExpectQuery(regexp.QuoteMeta("SELECT profile.name,profile.description,profile.status,revision.content_digest")).
			WithArgs(testNamespaceID, definition.id).
			WillReturnRows(sqlmock.NewRows([]string{
				"name", "description", "status", "content_digest",
			}).AddRow(definition.input.Name, definition.input.Description, string(agentmanagement.StatusActive), digest[:]))
		mock.ExpectQuery(regexp.QuoteMeta("SELECT count(*) FROM agent_profile_skills")).
			WithArgs(testNamespaceID, definition.id).
			WillReturnRows(sqlmock.NewRows([]string{"count"}).AddRow(len(definition.input.Skills)))
		for ordinal, skill := range definition.input.Skills {
			mock.ExpectQuery(regexp.QuoteMeta("SELECT skill_id::text,skill_revision")).
				WithArgs(testNamespaceID, definition.id, ordinal).
				WillReturnRows(sqlmock.NewRows([]string{"skill_id", "skill_revision"}).AddRow(skill.ID, skill.Revision))
		}
		for _, mode := range definition.input.DefaultForModes {
			mock.ExpectExec(regexp.QuoteMeta("INSERT INTO agent_profile_defaults")).
				WithArgs(testNamespaceID, mode, definition.id, now).
				WillReturnResult(sqlmock.NewResult(0, 0))
		}
	}
	mock.ExpectCommit()
}

func TestRetryDefaultMutationRetriesSerializableConflict(t *testing.T) {
	attempts := 0
	err := retryDefaultMutation(context.Background(), func() error {
		attempts++
		if attempts < defaultMutationAttempts {
			return agentmanagement.ErrConflict
		}
		return nil
	})
	if err != nil || attempts != defaultMutationAttempts {
		t.Fatalf("retryDefaultMutation() = (%v, %d attempts), want (nil, %d)", err, attempts, defaultMutationAttempts)
	}
}

func TestRetryDefaultMutationStopsOnPermanentFailure(t *testing.T) {
	permanent := errors.New("permanent failure")
	attempts := 0
	err := retryDefaultMutation(context.Background(), func() error {
		attempts++
		return permanent
	})
	if !errors.Is(err, permanent) || attempts != 1 {
		t.Fatalf("retryDefaultMutation() = (%v, %d attempts), want permanent failure after one attempt", err, attempts)
	}
}

func TestRetryDefaultMutationBoundsPersistentConflict(t *testing.T) {
	attempts := 0
	err := retryDefaultMutation(context.Background(), func() error {
		attempts++
		return agentmanagement.ErrConflict
	})
	if !errors.Is(err, agentmanagement.ErrConflict) || attempts != defaultMutationAttempts {
		t.Fatalf("retryDefaultMutation() = (%v, %d attempts), want conflict after %d", err, attempts, defaultMutationAttempts)
	}
}
