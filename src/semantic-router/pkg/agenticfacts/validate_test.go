package agenticfacts

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
	"time"
)

// fixedNow keeps expiry arithmetic reproducible and free of monotonic clock
// readings.
var fixedNow = time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC)

// validEnvelope returns the smallest envelope that passes validation. Tests
// mutate one key at a time so each case isolates a single rule.
func validEnvelope() map[string]any {
	return map[string]any{
		"version":    SchemaVersion,
		"expires_at": fixedNow.Add(time.Minute).Format(time.RFC3339),
	}
}

func encode(t *testing.T, envelope map[string]any) []byte {
	t.Helper()
	raw, err := json.Marshal(envelope)
	if err != nil {
		t.Fatalf("marshal fixture: %v", err)
	}
	return raw
}

// requireRejection asserts that validation failed with exactly one named
// rejection and produced no facts.
func requireRejection(t *testing.T, result Result, field string, reason string) {
	t.Helper()
	if result.Accepted != nil {
		t.Fatalf("rejected envelope produced facts: %#v", result.Accepted)
	}
	if len(result.Rejections) != 1 {
		t.Fatalf("want exactly one rejection, got %#v", result.Rejections)
	}
	got := result.Rejections[0]
	if got.Field != field || got.Reason != reason {
		t.Fatalf("want %q/%q, got %q/%q", field, reason, got.Field, got.Reason)
	}
}

func requireAccepted(t *testing.T, result Result) *Accepted {
	t.Helper()
	if len(result.Rejections) > 0 {
		t.Fatalf("unexpected rejections: %#v", result.Rejections)
	}
	if result.Accepted == nil {
		t.Fatal("expected accepted facts, got nil")
	}
	return result.Accepted
}

// canonicalCapabilityNames lists distinct protocol capabilities the validator
// accepts. It is longer than the default MaxCapabilities so cap tests can go one
// past the bound with names that would otherwise be valid.
var canonicalCapabilityNames = []string{
	"text", "image_input", "image_output", "audio_input", "audio_output",
	"video_input", "video_output", "file_input", "file_output", "tools",
	"parallel_tools", "reasoning", "structured_json", "strict_json_schema",
	"strict_tool_schema", "streaming", "stop_sequences",
}

// namedCapabilities returns n distinct canonical capability names.
func namedCapabilities(n int) []string {
	return append([]string(nil), canonicalCapabilityNames[:n]...)
}

func TestValidateAcceptsCompleteEnvelope(t *testing.T) {
	expiresAt := fixedNow.Add(time.Minute).Format(time.RFC3339)

	envelope := map[string]any{
		"version": SchemaVersion,
		"lineage": map[string]any{
			"root_invocation_id":   "inv-root",
			"parent_invocation_id": "inv-parent",
			"depth":                2,
		},
		"delegated_role": "security_review",
		"task_phase":     "execute",
		"budget": map[string]any{
			"remaining_tokens":  12000,
			"remaining_time_ms": 30000,
			"remaining_cost":    1.5,
		},
		"required_capabilities": []string{"Tools", "IMAGE_INPUT", "tools"},
		"context_portability":   "STICKY",
		"trust_boundary": map[string]any{
			"tenant":    "acme",
			"residency": "eu",
			"label":     "confidential",
		},
		"expires_at": expiresAt,
	}

	accepted := requireAccepted(t, Validate(encode(t, envelope), DefaultBounds(), fixedNow))

	if accepted.RootInvocationID != "inv-root" || accepted.ParentInvocationID != "inv-parent" {
		t.Fatalf("lineage identifiers not carried: %#v", accepted.acceptedScalars)
	}
	if accepted.Depth != 2 {
		t.Fatalf("want depth 2, got %d", accepted.Depth)
	}
	if accepted.DelegatedRole != "security_review" || accepted.TaskPhase != "execute" {
		t.Fatalf("role or phase not carried: %#v", accepted.acceptedScalars)
	}

	if !accepted.RemainingTokensKnown || accepted.RemainingTokens != 12000 {
		t.Fatalf("remaining tokens not carried: %#v", accepted.acceptedScalars)
	}
	if !accepted.RemainingTimeMsKnown || accepted.RemainingTimeMs != 30000 {
		t.Fatalf("remaining time not carried: %#v", accepted.acceptedScalars)
	}
	if !accepted.RemainingCostKnown || accepted.RemainingCost != 1.5 {
		t.Fatalf("remaining cost not carried: %#v", accepted.acceptedScalars)
	}

	if accepted.ContextPortability != ContextPortabilitySticky {
		t.Fatalf("want portability %q, got %q", ContextPortabilitySticky, accepted.ContextPortability)
	}
	if accepted.Tenant != "acme" || accepted.Residency != "eu" || accepted.TrustLabel != "confidential" {
		t.Fatalf("trust boundary not carried: %#v", accepted.acceptedScalars)
	}

	if !accepted.ExpiresAtKnown {
		t.Fatal("expiry not marked known")
	}
	wantExpiry, err := time.Parse(time.RFC3339, expiresAt)
	if err != nil {
		t.Fatalf("parse expected expiry: %v", err)
	}
	if !accepted.ExpiresAt.Equal(wantExpiry) {
		t.Fatalf("want expiry %s, got %s", wantExpiry, accepted.ExpiresAt)
	}
}

func TestValidateTreatsAbsentEnvelopeAsNoFacts(t *testing.T) {
	for name, raw := range map[string][]byte{
		"nil":   nil,
		"empty": {},
	} {
		t.Run(name, func(t *testing.T) {
			result := Validate(raw, DefaultBounds(), fixedNow)
			if result.Accepted != nil {
				t.Fatalf("absent envelope produced facts: %#v", result.Accepted)
			}
			if len(result.Rejections) != 0 {
				t.Fatalf("absent envelope produced diagnostics: %#v", result.Rejections)
			}
			if result.Rejected() || result.HasFacts() {
				t.Fatalf("want a neutral result, got %#v", result)
			}
		})
	}
}

// TestValidateStopsBeforeFieldChecks covers the three failures that end
// validation because nothing after them can be interpreted.
func TestValidateStopsBeforeFieldChecks(t *testing.T) {
	oversized := validEnvelope()
	oversized["delegated_role"] = strings.Repeat("a", 9000)

	missingVersion := validEnvelope()
	delete(missingVersion, "version")

	wrongVersion := validEnvelope()
	wrongVersion["version"] = "2"

	tests := []struct {
		name   string
		raw    []byte
		field  string
		reason string
	}{
		{"oversized", encode(t, oversized), "", ReasonTooLarge},
		{"malformed json", []byte(`{"version":`), "", ReasonMalformed},
		{"missing version", encode(t, missingVersion), "version", ReasonMissing},
		{"unsupported version", encode(t, wrongVersion), "version", ReasonUnsupportedVersion},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			requireRejection(t, Validate(test.raw, DefaultBounds(), fixedNow), test.field, test.reason)
		})
	}
}

func TestValidateExpiry(t *testing.T) {
	missing := validEnvelope()
	delete(missing, "expires_at")

	tests := []struct {
		name      string
		expiresAt any
		reason    string // empty means the envelope must be accepted
	}{
		{"unparseable", "not-a-time", ReasonMalformed},
		{"stale", fixedNow.Add(-10 * time.Minute).Format(time.RFC3339), ReasonExpired},
		{"stale within clock skew", fixedNow.Add(-3 * time.Second).Format(time.RFC3339), ""},
		{"beyond max lifetime", fixedNow.Add(time.Hour).Format(time.RFC3339), ReasonTooLong},
		{"just inside max lifetime", fixedNow.Add(4*time.Minute + 59*time.Second).Format(time.RFC3339), ""},
	}

	t.Run("absent", func(t *testing.T) {
		requireRejection(t, Validate(encode(t, missing), DefaultBounds(), fixedNow), "expires_at", ReasonMissing)
	})

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			envelope := validEnvelope()
			envelope["expires_at"] = test.expiresAt

			result := Validate(encode(t, envelope), DefaultBounds(), fixedNow)
			if test.reason == "" {
				requireAccepted(t, result)
				return
			}
			requireRejection(t, result, "expires_at", test.reason)
		})
	}
}

func TestValidateLineage(t *testing.T) {
	tests := []struct {
		name    string
		lineage map[string]any
		field   string
		reason  string // empty means the envelope must be accepted
	}{
		{
			name:    "negative depth",
			lineage: map[string]any{"depth": -1},
			field:   "lineage.depth",
			reason:  ReasonMalformed,
		},
		{
			name: "depth over bound",
			lineage: map[string]any{
				"root_invocation_id":   "inv-root",
				"parent_invocation_id": "inv-parent",
				"depth":                17,
			},
			field:  "lineage.depth",
			reason: ReasonTooDeep,
		},
		{
			name: "depth at bound",
			lineage: map[string]any{
				"root_invocation_id":   "inv-root",
				"parent_invocation_id": "inv-parent",
				"depth":                16,
			},
		},
		{
			name: "delegated without parent",
			lineage: map[string]any{
				"root_invocation_id": "inv-root",
				"depth":              2,
			},
			field:  "lineage.parent_invocation_id",
			reason: ReasonConflicting,
		},
		{
			name:    "parent without root",
			lineage: map[string]any{"parent_invocation_id": "inv-parent"},
			field:   "lineage.root_invocation_id",
			reason:  ReasonConflicting,
		},
		{
			name:    "identifier over length",
			lineage: map[string]any{"root_invocation_id": strings.Repeat("a", 200)},
			field:   "lineage.root_invocation_id",
			reason:  ReasonTooLong,
		},
		{
			// A parent that is present but too long is one problem, not two:
			// it must not also be reported as missing for a delegated depth.
			name: "parent identifier over length with depth",
			lineage: map[string]any{
				"root_invocation_id":   "inv-root",
				"parent_invocation_id": strings.Repeat("p", 200),
				"depth":                2,
			},
			field:  "lineage.parent_invocation_id",
			reason: ReasonTooLong,
		},
		{
			// Same for a root that is present but too long next to a parent.
			name: "root identifier over length with parent",
			lineage: map[string]any{
				"root_invocation_id":   strings.Repeat("r", 200),
				"parent_invocation_id": "inv-parent",
				"depth":                1,
			},
			field:  "lineage.root_invocation_id",
			reason: ReasonTooLong,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			envelope := validEnvelope()
			envelope["lineage"] = test.lineage

			result := Validate(encode(t, envelope), DefaultBounds(), fixedNow)
			if test.reason == "" {
				requireAccepted(t, result)
				return
			}
			requireRejection(t, result, test.field, test.reason)
		})
	}
}

func TestValidateRole(t *testing.T) {
	t.Run("unknown portability", func(t *testing.T) {
		envelope := validEnvelope()
		envelope["context_portability"] = "maybe"
		requireRejection(t, Validate(encode(t, envelope), DefaultBounds(), fixedNow),
			"context_portability", ReasonMalformed)
	})

	t.Run("portability is case insensitive", func(t *testing.T) {
		envelope := validEnvelope()
		envelope["context_portability"] = "Portable"

		accepted := requireAccepted(t, Validate(encode(t, envelope), DefaultBounds(), fixedNow))
		if accepted.ContextPortability != ContextPortabilityPortable {
			t.Fatalf("want %q, got %q", ContextPortabilityPortable, accepted.ContextPortability)
		}
	})

	t.Run("role over length", func(t *testing.T) {
		envelope := validEnvelope()
		envelope["delegated_role"] = strings.Repeat("a", 200)
		requireRejection(t, Validate(encode(t, envelope), DefaultBounds(), fixedNow),
			"delegated_role", ReasonTooLong)
	})
}

// TestValidateBudgetDistinguishesAbsentFromZero is the reason the Accepted
// scalars carry XxxKnown companions: a zero budget is a real, exhausted budget
// and must not read as an absent one.
func TestValidateBudgetDistinguishesAbsentFromZero(t *testing.T) {
	t.Run("zero is known", func(t *testing.T) {
		envelope := validEnvelope()
		envelope["budget"] = map[string]any{"remaining_tokens": 0}

		accepted := requireAccepted(t, Validate(encode(t, envelope), DefaultBounds(), fixedNow))
		if !accepted.RemainingTokensKnown {
			t.Fatal("zero remaining tokens must be marked known")
		}
		if accepted.RemainingTokens != 0 {
			t.Fatalf("want 0 remaining tokens, got %d", accepted.RemainingTokens)
		}
	})

	t.Run("absent is unknown", func(t *testing.T) {
		accepted := requireAccepted(t, Validate(encode(t, validEnvelope()), DefaultBounds(), fixedNow))
		if accepted.RemainingTokensKnown {
			t.Fatal("absent budget must not be marked known")
		}
	})
}

func TestValidateBudget(t *testing.T) {
	tests := []struct {
		name   string
		budget map[string]any
		field  string
		reason string
	}{
		{
			name:   "negative tokens",
			budget: map[string]any{"remaining_tokens": -1},
			field:  "budget.remaining_tokens",
			reason: ReasonMalformed,
		},
		{
			name:   "negative time",
			budget: map[string]any{"remaining_time_ms": -1},
			field:  "budget.remaining_time_ms",
			reason: ReasonMalformed,
		},
		{
			name:   "negative cost",
			budget: map[string]any{"remaining_cost": -0.5},
			field:  "budget.remaining_cost",
			reason: ReasonMalformed,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			envelope := validEnvelope()
			envelope["budget"] = test.budget
			requireRejection(t, Validate(encode(t, envelope), DefaultBounds(), fixedNow),
				test.field, test.reason)
		})
	}
}

// TestValidateRejectsOverCapListsRatherThanTruncating pins the rule that a
// cardinality bound is checked against what the caller actually sent. Silently
// keeping the first N entries would narrow the request differently from what
// the caller asked for, without telling anyone.
func TestValidateRejectsOverCapListsRatherThanTruncating(t *testing.T) {
	t.Run("distinct capabilities over cap", func(t *testing.T) {
		envelope := validEnvelope()
		envelope["required_capabilities"] = namedCapabilities(17)
		requireRejection(t, Validate(encode(t, envelope), DefaultBounds(), fixedNow),
			"required_capabilities", ReasonTooMany)
	})

	// Guards the ordering inside checkCapabilities: if dedupSorted ever runs
	// before the cardinality check, forty duplicates collapse to one entry and
	// this envelope wrongly passes.
	t.Run("duplicate capabilities over cap", func(t *testing.T) {
		duplicates := make([]string, 0, 40)
		for i := 0; i < 40; i++ {
			duplicates = append(duplicates, "image_input")
		}

		envelope := validEnvelope()
		envelope["required_capabilities"] = duplicates
		requireRejection(t, Validate(encode(t, envelope), DefaultBounds(), fixedNow),
			"required_capabilities", ReasonTooMany)
	})

	t.Run("capabilities at cap", func(t *testing.T) {
		envelope := validEnvelope()
		envelope["required_capabilities"] = namedCapabilities(16)

		accepted := requireAccepted(t, Validate(encode(t, envelope), DefaultBounds(), fixedNow))
		if len(accepted.RequiredCapabilities) != 16 {
			t.Fatalf("want 16 capabilities, got %#v", accepted.RequiredCapabilities)
		}
	})
}

func TestValidateListEntries(t *testing.T) {
	tests := []struct {
		name   string
		key    string
		values []string
		reason string
	}{
		{"empty capability", "required_capabilities", []string{"image_input", ""}, ReasonMalformed},
		{"over-long capability", "required_capabilities", []string{strings.Repeat("a", 200)}, ReasonTooLong},
		{"unknown capability", "required_capabilities", []string{"tools", "banana"}, ReasonMalformed},
		{"model card alias", "required_capabilities", []string{"vision"}, ReasonMalformed},
		{"tool_use alias", "required_capabilities", []string{"tool_use"}, ReasonMalformed},
		{"chat alias", "required_capabilities", []string{"chat"}, ReasonMalformed},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			envelope := validEnvelope()
			envelope[test.key] = test.values
			requireRejection(t, Validate(encode(t, envelope), DefaultBounds(), fixedNow),
				test.key, test.reason)
		})
	}
}

// TestValidateNormalizesLists pins capability normalization: capabilities are
// symbolic tokens matched against model card metadata, so they fold to lower
// case, are trimmed, deduplicated, and sorted.
func TestValidateNormalizesLists(t *testing.T) {
	envelope := validEnvelope()
	envelope["required_capabilities"] = []string{"Tools", "IMAGE_INPUT", "tools", " image_input "}

	accepted := requireAccepted(t, Validate(encode(t, envelope), DefaultBounds(), fixedNow))

	wantCapabilities := []string{"image_input", "tools"}
	if !reflect.DeepEqual(accepted.RequiredCapabilities, wantCapabilities) {
		t.Fatalf("want %#v, got %#v", wantCapabilities, accepted.RequiredCapabilities)
	}
}

// TestValidateNeverAcceptsWithRejections pins the invariant the whole failure
// policy rests on: a rejected envelope contributes nothing to selection.
func TestValidateNeverAcceptsWithRejections(t *testing.T) {
	oversized := validEnvelope()
	oversized["delegated_role"] = strings.Repeat("a", 9000)

	staleExpiry := validEnvelope()
	staleExpiry["expires_at"] = fixedNow.Add(-time.Hour).Format(time.RFC3339)

	badLineage := validEnvelope()
	badLineage["lineage"] = map[string]any{"depth": 99}

	badPortability := validEnvelope()
	badPortability["context_portability"] = "maybe"

	tooManyCapabilities := validEnvelope()
	tooManyCapabilities["required_capabilities"] = namedCapabilities(17)

	fixtures := map[string][]byte{
		"oversized":             encode(t, oversized),
		"malformed json":        []byte(`{`),
		"stale expiry":          encode(t, staleExpiry),
		"bad lineage":           encode(t, badLineage),
		"bad portability":       encode(t, badPortability),
		"too many capabilities": encode(t, tooManyCapabilities),
	}

	for name, raw := range fixtures {
		t.Run(name, func(t *testing.T) {
			result := Validate(raw, DefaultBounds(), fixedNow)
			if !result.Rejected() {
				t.Fatalf("expected rejection, got %#v", result)
			}
			if result.Accepted != nil {
				t.Fatalf("rejected envelope produced facts: %#v", result.Accepted)
			}
			if result.HasFacts() {
				t.Fatal("rejected envelope reported facts")
			}
		})
	}
}

// TestValidateIsDeterministic covers the reason sortRejections exists: the same
// bytes must produce the same Result, in the same order, every time.
func TestValidateIsDeterministic(t *testing.T) {
	envelope := validEnvelope()
	envelope["context_portability"] = "maybe"
	envelope["delegated_role"] = strings.Repeat("a", 200)
	envelope["budget"] = map[string]any{"remaining_tokens": -1}

	raw := encode(t, envelope)

	first := Validate(raw, DefaultBounds(), fixedNow)
	second := Validate(raw, DefaultBounds(), fixedNow)

	if !reflect.DeepEqual(first, second) {
		t.Fatalf("validation is not reproducible:\nfirst  %#v\nsecond %#v", first, second)
	}

	wantFields := []string{"budget.remaining_tokens", "context_portability", "delegated_role"}
	gotFields := make([]string, 0, len(first.Rejections))
	for _, rejection := range first.Rejections {
		gotFields = append(gotFields, rejection.Field)
	}
	if !reflect.DeepEqual(gotFields, wantFields) {
		t.Fatalf("rejections not sorted by field: want %#v, got %#v", wantFields, gotFields)
	}
}

// TestZeroBoundsApplyDefaultCaps covers withDefaults: an operator who forgets to
// populate Bounds gets the default caps rather than an unbounded validator.
func TestZeroBoundsApplyDefaultCaps(t *testing.T) {
	envelope := validEnvelope()
	envelope["required_capabilities"] = namedCapabilities(17)

	requireRejection(t, Validate(encode(t, envelope), Bounds{}, fixedNow),
		"required_capabilities", ReasonTooMany)
}

// TestZeroClockSkewIsHonoured documents the one field withDefaults leaves alone:
// zero skew is a deliberate choice, so Bounds{} is stricter than DefaultBounds.
func TestZeroClockSkewIsHonoured(t *testing.T) {
	envelope := validEnvelope()
	envelope["expires_at"] = fixedNow.Add(-3 * time.Second).Format(time.RFC3339)
	raw := encode(t, envelope)

	requireAccepted(t, Validate(raw, DefaultBounds(), fixedNow))
	requireRejection(t, Validate(raw, Bounds{}, fixedNow), "expires_at", ReasonExpired)
}
