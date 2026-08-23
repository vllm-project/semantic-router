package usageledger

import (
	"errors"
	"strings"
	"testing"
	"time"
)

func TestUsageQueryAcceptsStableRoutingUIDs(t *testing.T) {
	query := UsageQuery{
		NamespaceID: "11111111-1111-4111-8111-111111111111",
		Start:       time.Date(2026, time.August, 22, 0, 0, 0, 0, time.UTC),
		End:         time.Date(2026, time.August, 22, 1, 0, 0, 0, time.UTC),
		Grain:       GrainMinute,
		Visibility:  QueryVisibility{All: true},
		Filters: UsageFilters{
			EntrypointID:   "public-chat",
			RecipeID:       "balanced-routing",
			LogicalModelID: "frontier-reasoning",
		},
	}
	if err := validateUsageQuery(query); err != nil {
		t.Fatalf("stable routing UIDs were rejected: %v", err)
	}
}

func TestUsageQueryUsesFinerRollupsForLocalCalendarBuckets(t *testing.T) {
	query := UsageQuery{
		NamespaceID: "11111111-1111-4111-8111-111111111111",
		Start:       time.Date(2026, time.August, 22, 0, 0, 0, 0, time.UTC),
		End:         time.Date(2026, time.August, 24, 0, 0, 0, 0, time.UTC),
		Grain:       GrainDay,
		TimeZone:    "Asia/Shanghai",
		Visibility:  QueryVisibility{All: true},
	}
	if err := validateUsageQuery(query); err != nil {
		t.Fatalf("valid IANA time zone was rejected: %v", err)
	}
	table, unit := seriesSource(query.Grain, query.TimeZone)
	if table != "usage_rollup_1h" || unit != "day" {
		t.Fatalf("local day source = (%q, %q), want hourly rows regrouped by calendar day", table, unit)
	}
	query.TimeZone = "Not/AZone"
	if err := validateUsageQuery(query); !errors.Is(err, ErrInvalidQuery) {
		t.Fatalf("invalid time zone error = %v, want ErrInvalidQuery", err)
	}
}

func TestUsageQueryRejectsMalformedRoutingUIDs(t *testing.T) {
	base := UsageQuery{
		NamespaceID: "11111111-1111-4111-8111-111111111111",
		Start:       time.Date(2026, time.August, 22, 0, 0, 0, 0, time.UTC),
		End:         time.Date(2026, time.August, 22, 1, 0, 0, 0, time.UTC),
		Grain:       GrainMinute,
		Visibility:  QueryVisibility{All: true},
	}
	for name, mutate := range map[string]func(*UsageQuery){
		"entrypoint": func(query *UsageQuery) { query.Filters.EntrypointID = "contains space" },
		"recipe":     func(query *UsageQuery) { query.Filters.RecipeID = strings.Repeat("x", 129) },
		"model":      func(query *UsageQuery) { query.Filters.LogicalModelID = "?secret=value" },
	} {
		t.Run(name, func(t *testing.T) {
			query := base
			mutate(&query)
			if err := validateUsageQuery(query); err == nil {
				t.Fatal("malformed routing UID was accepted")
			}
		})
	}
}

func TestRawLogFiltersUseTextForRoutingUIDs(t *testing.T) {
	query := LogQuery{
		NamespaceID: "11111111-1111-4111-8111-111111111111",
		Start:       time.Date(2026, time.August, 22, 0, 0, 0, 0, time.UTC),
		End:         time.Date(2026, time.August, 22, 1, 0, 0, 0, time.UTC),
		Visibility:  QueryVisibility{All: true},
		Filters: UsageFilters{
			EntrypointID: "public-chat",
			RecipeID:     "balanced-routing",
		},
	}
	where, _ := rawLogWhere(query)
	if strings.Contains(where, "entrypoint_id = $6::uuid") || strings.Contains(where, "recipe_id = $7::uuid") {
		t.Fatalf("routing UID filters still use UUID casts: %s", where)
	}
	if !strings.Contains(where, "entrypoint_id = $6::text") || !strings.Contains(where, "recipe_id = $7::text") {
		t.Fatalf("routing UID filters are not typed as text: %s", where)
	}
}

func TestUsageQueriesRequireServerAuthorizedVisibility(t *testing.T) {
	query := UsageQuery{
		NamespaceID: "11111111-1111-4111-8111-111111111111",
		Start:       time.Date(2026, time.August, 22, 0, 0, 0, 0, time.UTC),
		End:         time.Date(2026, time.August, 22, 1, 0, 0, 0, time.UTC),
		Grain:       GrainMinute,
	}
	if err := validateUsageQuery(query); err == nil {
		t.Fatal("usage query without an authorized result scope was accepted")
	} else if !errors.Is(err, ErrInvalidQuery) {
		t.Fatalf("missing visibility error = %v, want ErrInvalidQuery", err)
	}
	query.Visibility = QueryVisibility{TeamIDs: []string{"22222222-2222-4222-8222-222222222222"}}
	if err := validateUsageQuery(query); err != nil {
		t.Fatalf("narrow authorized visibility was rejected: %v", err)
	}
	where, _ := rollupWhere(query, RollupRequest)
	if !strings.Contains(where, "dimensions->>'teamId' = ANY(") {
		t.Fatalf("authorized Team scope is absent from SQL: %s", where)
	}
}

func TestRawLogVisibilityIsBoundIntoCursorAndSQL(t *testing.T) {
	base := LogQuery{
		NamespaceID: "11111111-1111-4111-8111-111111111111",
		Start:       time.Date(2026, time.August, 22, 0, 0, 0, 0, time.UTC),
		End:         time.Date(2026, time.August, 22, 1, 0, 0, 0, time.UTC),
		Visibility: QueryVisibility{UserIDs: []string{
			"22222222-2222-4222-8222-222222222222",
		}},
	}
	where, _ := rawLogWhere(base)
	if !strings.Contains(where, "user_id = ANY(") {
		t.Fatalf("authorized User scope is absent from raw log SQL: %s", where)
	}
	other := base
	other.Visibility = QueryVisibility{UserIDs: []string{"33333333-3333-4333-8333-333333333333"}}
	if logFilterDigest(base) == logFilterDigest(other) {
		t.Fatal("request-log cursor digest did not bind authorized visibility")
	}
}

func TestRawLogPageQueryPrunesPartitionsAndUsesBoundedKeyset(t *testing.T) {
	query := LogQuery{
		NamespaceID: "11111111-1111-4111-8111-111111111111",
		Start:       time.Date(2026, time.August, 22, 0, 0, 0, 0, time.UTC),
		End:         time.Date(2026, time.August, 22, 1, 0, 0, 0, time.UTC),
		PageSize:    50,
		Visibility:  QueryVisibility{All: true},
	}
	statement, args := rawLogPageQuery(query, &logCursor{
		OccurredAt: query.End.Add(-time.Second).UnixNano(),
		EventID:    "22222222-2222-4222-8222-222222222222",
	})
	for _, contract := range []string{
		"event_date >= $4::date", "event_date < $5::date",
		"(occurred_at, event_id) < ($6, $7::uuid)",
		"ORDER BY occurred_at DESC, event_id DESC LIMIT $8",
	} {
		if !strings.Contains(statement, contract) {
			t.Fatalf("raw log query lacks %q: %s", contract, statement)
		}
	}
	if len(args) != 8 || args[7] != 51 {
		t.Fatalf("raw log query args = %#v, want bounded page-size-plus-one", args)
	}
}
