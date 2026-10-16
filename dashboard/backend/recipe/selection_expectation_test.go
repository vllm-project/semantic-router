package recipe

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestProbeSelectionExpectationAdmissionAndVariantOverride(t *testing.T) {
	directory := writeManagedRecipe(t)
	content, err := os.ReadFile(filepath.Join(directory, "probes.yaml"))
	require.NoError(t, err)
	manifestText := strings.Replace(string(content), "    expected_algorithm: static", "    expected_algorithm: static\n    expected_selection_status: selected", 1)
	manifestText = strings.Replace(manifestText, "      - id: messages-a", "      - id: messages-a\n        expected_selection_status: unavailable", 1)
	manifest, err := decodeProbes([]byte(manifestText))
	require.NoError(t, err)
	probes, _ := flattenProbes(manifest)
	require.Equal(t, "selected", probes[0].Expected.SelectionStatus)
	require.Equal(t, "unavailable", probes[1].Expected.SelectionStatus)

	for _, value := range []string{"unknown", "''", "'unavailable '"} {
		for _, field := range []string{"    expected_selection_status: selected", "        expected_selection_status: unavailable"} {
			t.Run(field+value, func(t *testing.T) {
				invalid := strings.Replace(manifestText, field, strings.Split(field, ":")[0]+": "+value, 1)
				_, err := decodeProbes([]byte(invalid))
				require.ErrorContains(t, err, "expected_selection_status")
			})
		}
	}
}

func TestCompareEvalResponseExplicitUnavailableSelection(t *testing.T) {
	probe, raw := selectionContractFixture("static", "", "unavailable", "", []string{"candidate-a"})
	probe.Expected.SelectionStatus = "unavailable"
	probe.Expected.Signals = map[string][]string{"context": {"long"}}
	var response evalResponse
	require.NoError(t, json.Unmarshal(raw, &response))
	response.SelectionReason = "input exceeds all backend context capacities"
	response.DecisionResult.MatchedSignals = map[string][]string{"context": {"long"}}

	for _, test := range []struct {
		name          string
		mutate        func(*evalResponse)
		wantSelection bool
		wantSignals   bool
	}{
		{"capacity rejection", func(*evalResponse) {}, true, true},
		{"wrong status", func(r *evalResponse) {
			r.SelectionStatus = "selected"
			r.SelectedModel = "candidate-a"
			r.SelectionMethod = "static"
		}, false, true},
		{"fabricated selection", func(r *evalResponse) { r.SelectedModel = "candidate-a" }, false, true},
		{"missing reason", func(r *evalResponse) { r.SelectionReason = "" }, false, true},
		{"wrong signals", func(r *evalResponse) { r.DecisionResult.MatchedSignals = nil }, true, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			modified := response
			test.mutate(&modified)
			encoded, err := json.Marshal(modified)
			require.NoError(t, err)
			actual, checks, failures, err := compareEvalResponse(encoded, probe, []ProbeDetail{probe})
			require.NoError(t, err)
			require.Equal(t, test.wantSelection, checks.Selection, "%v", failures)
			require.Equal(t, test.wantSignals, checks.Signals, "%v", failures)
			if test.wantSelection && test.wantSignals {
				require.Empty(t, failures)
				require.Empty(t, actual.Model)
				require.Equal(t, "unavailable", actual.SelectionStatus)
				require.NotEmpty(t, actual.SelectionReason)
			} else {
				require.NotEmpty(t, failures)
			}
		})
	}

	probe.Expected.SelectionStatus = ""
	encoded, err := json.Marshal(response)
	require.NoError(t, err)
	_, checks, _, err := compareEvalResponse(encoded, probe, []ProbeDetail{probe})
	require.NoError(t, err)
	require.False(t, checks.Selection, "omitted expectation must retain the default positive assertion")
}
