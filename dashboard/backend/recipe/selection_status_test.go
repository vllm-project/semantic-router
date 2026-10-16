package recipe

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestProbeSelectionStatusDecodeAndMaterialization(t *testing.T) {
	directory := writeManagedRecipe(t)
	source, readErr := os.ReadFile(filepath.Join(directory, "probes.yaml"))
	require.NoError(t, readErr)
	legacy, decodeErr := decodeProbes(source)
	require.NoError(t, decodeErr)
	legacyProbes, _ := flattenProbes(legacy)
	for _, status := range []string{"unavailable", "not_required"} {
		t.Run(status, func(t *testing.T) {
			manifest, err := decodeProbes(withExpectedSelectionStatus(source, status))
			require.NoError(t, err)
			probes, _ := flattenProbes(manifest)
			require.Len(t, probes, len(legacyProbes))
			for index, probe := range probes {
				require.Equal(t, legacyProbes[index].ID, probe.ID)
				require.Equal(t, status, cloneExpectedAssertions(probe.Expected).SelectionStatus)
				before, err := materializeEvalRequest(legacyProbes[index])
				require.NoError(t, err)
				after, err := materializeEvalRequest(probe)
				require.NoError(t, err)
				// Expectations belong to the validator, never the model request.
				require.Equal(t, before, after)
			}
		})
	}
	for _, value := range []string{"invented", "\"\"", "[]"} {
		_, err := decodeProbes(withExpectedSelectionStatus(source, value))
		require.Error(t, err, value)
	}
	unknown := strings.Replace(string(source), "expected_algorithm: static", "unexpected_selection_status: unavailable", 1)
	_, unknownErr := decodeProbes([]byte(unknown))
	require.ErrorContains(t, unknownErr, "field unexpected_selection_status not found")
}

func withExpectedSelectionStatus(source []byte, value string) []byte {
	return []byte(strings.Replace(string(source), "    expected_algorithm: static", "    expected_algorithm: static\n    expected_selection_status: "+value, 1))
}

func TestServiceValidationEnforcesExplicitSelectionStatus(t *testing.T) {
	for _, status := range []string{"unavailable", "not_required"} {
		t.Run(status, func(t *testing.T) {
			directory := writeManagedRecipe(t)
			source, err := os.ReadFile(filepath.Join(directory, "probes.yaml"))
			require.NoError(t, err)
			writeFile(t, directory, "probes.yaml", string(withExpectedSelectionStatus(source, status)))
			actualStatus := status
			service := NewService(Options{Directory: directory, Evaluator: evaluatorFunc(func(_ context.Context, request EvalRequest) (json.RawMessage, error) {
				require.Equal(t, "vllm-sr/mom-test-v1", request.Model)
				return json.Marshal(map[string]any{
					"requested_model": request.Model, "recipe": "balanced",
					"routing_decision": "decision-a", "selection_status": actualStatus,
					"selection_method": map[string]string{"not_required": "fast_response"}[status], "selection_reason": "Synthetic selection outcome.",
					"decision_result": map[string]any{
						"decision_name": "decision-a", "algorithm": "static", "plugins": []string{"tools"},
						"matched_signals": map[string][]string{"keywords": {"signal-a"}},
					},
					"recommended_models": []string{"leaf-a"},
					"eval_trace":         []map[string]any{{"decision_name": "decision-a", "matched": true}},
				})
			})})
			result, err := service.Validate(context.Background(), "lane-a", "variant-a")
			require.NoError(t, err)
			require.True(t, result.Passed, result.Failures)
			require.True(t, result.Checks.Selection)
			require.Equal(t, status, result.Expected.SelectionStatus)
			require.Equal(t, status, result.Actual.SelectionStatus)
			require.Empty(t, result.Actual.Model)
			encoded, err := json.Marshal(result)
			require.NoError(t, err)
			var wire map[string]any
			require.NoError(t, json.Unmarshal(encoded, &wire))
			require.Equal(t, status, wire["expected"].(map[string]any)["selection_status"])
			actualStatus = "failed"
			result, err = service.Validate(context.Background(), "lane-a", "variant-a")
			require.NoError(t, err)
			require.False(t, result.Passed)
			require.False(t, result.Checks.Selection)
		})
	}
}

func TestExplicitSelectionStatusKeepsAllOtherAssertions(t *testing.T) {
	for _, status := range []string{"unavailable", "not_required"} {
		t.Run(status, func(t *testing.T) {
			probe := ProbeDetail{ProbeSummary: ProbeSummary{Expected: ExpectedAssertions{
				Decision: "route-a", SelectionStatus: status, Plugins: []string{"fast_response"},
				Signals: map[string][]string{"keyword": {"synthetic"}},
			}}}
			base := evalResponse{
				SelectionStatus: status, SelectionMethod: "fast_response", SelectionReason: "Synthetic reason.",
				Recipe: "default", RoutingDecision: "route-a",
				DecisionResult: evalDecisionResult{DecisionName: "route-a", Plugins: []string{"fast_response"}, MatchedSignals: map[string][]string{"keyword": {"synthetic"}}},
				EvalTrace:      []evalTrace{{DecisionName: "route-a", Matched: true}},
			}
			for name, mutate := range map[string]func(*evalResponse){
				"wrong status":    func(r *evalResponse) { r.SelectionStatus = "failed" },
				"selected model":  func(r *evalResponse) { r.SelectedModel = "fabricated" },
				"final model":     func(r *evalResponse) { r.FinalModel = "fabricated" },
				"missing reason":  func(r *evalResponse) { r.SelectionReason = " " },
				"wrong decision":  func(r *evalResponse) { r.RoutingDecision = "other" },
				"missing plugin":  func(r *evalResponse) { r.DecisionResult.Plugins = nil },
				"missing signal":  func(r *evalResponse) { r.DecisionResult.MatchedSignals = nil },
				"missing trace":   func(r *evalResponse) { r.EvalTrace = nil },
				"wrong algorithm": func(r *evalResponse) { r.DecisionResult.Algorithm = "fusion" },
			} {
				t.Run(name, func(t *testing.T) {
					response := base
					mutate(&response)
					currentProbe := probe
					if name == "wrong algorithm" {
						currentProbe.Expected.Algorithm = "static"
					}
					raw, err := json.Marshal(response)
					require.NoError(t, err)
					_, _, failures, err := compareEvalResponse(raw, currentProbe, []ProbeDetail{currentProbe})
					require.NoError(t, err)
					require.NotEmpty(t, failures)
				})
			}
			base.SelectionMethod = ""
			passed, _ := compareExpectedSelection(probe.Expected, base, nil)
			require.Equal(t, status == "unavailable", passed)
		})
	}
}
