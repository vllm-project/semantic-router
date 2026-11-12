package recipe

import (
	"context"
	"encoding/json"
	"maps"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

const errorProbeManifest = `schema_version: v1
name: explicit-errors
routing_assets: {yaml: config.yaml, dsl: recipe.dsl}
coverage: {}
decisions:
  - id: example
    expected_decision: route
    expected_algorithm: static
    expected_signal_errors: {reask:repeat: reask_evaluation_failed}
    variants:
      - id: inherited
        query: Complete input.
      - id: replaced
        query: Another complete input.
        expected_signal_errors: {projection:retry: projection_input_failed}
      - id: cleared
        query: Short input.
        expected_signal_errors: {}
`

func TestExpectedSignalErrorsAdmissionAndMaterialization(t *testing.T) {
	manifest, err := decodeProbes([]byte(errorProbeManifest))
	require.NoError(t, err)
	probes, _ := flattenProbes(manifest)
	require.Equal(t, map[string]string{"reask:repeat": "reask_evaluation_failed"}, probes[0].Expected.SignalErrors)
	require.Equal(t, map[string]string{"projection:retry": "projection_input_failed"}, probes[1].Expected.SignalErrors)
	require.Empty(t, probes[2].Expected.SignalErrors)
	cloned := cloneExpectedAssertions(probes[0].Expected)
	cloned.SignalErrors["reask:repeat"] = "changed"
	require.Equal(t, "reask_evaluation_failed", probes[0].Expected.SignalErrors["reask:repeat"])
	for _, probe := range probes {
		request, materializeErr := materializeEvalRequest(probe)
		require.NoError(t, materializeErr)
		encoded, marshalErr := json.Marshal(request)
		require.NoError(t, marshalErr)
		require.NotContains(t, string(encoded), "signal_errors", "assertions must not alter inference requests")
	}
	encoded, err := json.Marshal(probes[0].Expected)
	require.NoError(t, err)
	require.Contains(t, string(encoded), `"signal_errors":{"reask:repeat":"reask_evaluation_failed"}`)
}

func TestExpectedSignalErrorsRejectInvalidMaps(t *testing.T) {
	for _, value := range []string{
		"null", "[]", "ignore", "{'': failed}", "{' reask:repeat': failed}",
		"{reask:repeat: ''}", "{reask:repeat: 'failed '}", "{reask:repeat: true}",
		"{reask:repeat: 1}", "{reask:repeat: null}", "{reask:repeat: [failed]}",
		"{reask:repeat: failed, reask:repeat: other}",
	} {
		for _, field := range []string{
			"expected_signal_errors: {reask:repeat: reask_evaluation_failed}",
			"expected_signal_errors: {}",
		} {
			t.Run(field+value, func(t *testing.T) {
				invalid := strings.Replace(errorProbeManifest, field, "expected_signal_errors: "+value, 1)
				_, err := decodeProbes([]byte(invalid))
				require.ErrorContains(t, err, "expected_signal_errors")
			})
		}
	}
}

func TestExpectedSignalErrorsComparisonIsExactAndDefaultEmpty(t *testing.T) {
	expected := map[string]string{"reask:repeat": "reask_evaluation_failed", "projection:retry": "projection_input_failed"}
	for _, test := range []struct {
		name             string
		expected, actual map[string]string
		want             bool
	}{
		{"default absent", nil, nil, true},
		{"default empty", nil, map[string]string{}, true},
		{"default rejects errors", nil, expected, false},
		{"exact", expected, maps.Clone(expected), true},
		{"silent success", expected, nil, false},
		{"missing", expected, map[string]string{"reask:repeat": "reask_evaluation_failed"}, false},
		{"wrong code", expected, map[string]string{"reask:repeat": "wrong", "projection:retry": "projection_input_failed"}, false},
		{"extra", expected, map[string]string{"reask:repeat": "reask_evaluation_failed", "projection:retry": "projection_input_failed", "other": "failed"}, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			probe, raw := selectionContractFixture("static", "candidate-a", "selected", "static", []string{"candidate-a"})
			probe.Expected.SignalErrors = test.expected
			var response evalResponse
			require.NoError(t, json.Unmarshal(raw, &response))
			response.SignalErrors = test.actual
			raw, err := json.Marshal(response)
			require.NoError(t, err)
			actual, checks, failures, err := compareEvalResponse(raw, probe, []ProbeDetail{probe})
			require.NoError(t, err)
			require.Equal(t, test.want, checks.SignalErrors)
			require.Equal(t, test.want, len(failures) == 0, failures)
			require.Equal(t, test.actual, actual.SignalErrors)
		})
	}
}

func TestServiceValidationCarriesAndEnforcesSignalErrors(t *testing.T) {
	directory := writeManagedRecipe(t)
	writeFile(t, directory, "probes.yaml", strings.Replace(errorProbeManifest, "name: explicit-errors", "name: test-recipe", 1))
	errors := map[string]string{"reask:repeat": "reask_evaluation_failed"}
	decision := "route"
	service := NewService(Options{Directory: directory, Evaluator: evaluatorFunc(func(_ context.Context, request EvalRequest) (json.RawMessage, error) {
		response := rawValueResponse(.3)
		response["requested_model"] = request.Model
		response["routing_decision"] = decision
		response["signal_errors"] = errors
		return json.Marshal(response)
	})})
	result, err := service.Validate(context.Background(), "example", "inherited")
	require.NoError(t, err)
	require.True(t, result.Passed, result.Failures)
	require.Equal(t, errors, result.Expected.SignalErrors)
	require.Equal(t, errors, result.Actual.SignalErrors)
	encoded, err := json.Marshal(result)
	require.NoError(t, err)
	require.Equal(t, 2, strings.Count(string(encoded), `"signal_errors":{"reask:repeat":"reask_evaluation_failed"}`))
	delete(result.Expected.SignalErrors, "reask:repeat")
	decision = "wrong"
	result, err = service.Validate(context.Background(), "example", "inherited")
	require.NoError(t, err)
	require.False(t, result.Passed, "an expected error must not waive routing checks")
	require.True(t, result.Checks.SignalErrors)
	decision = "route"
	errors = nil
	result, err = service.Validate(context.Background(), "example", "inherited")
	require.NoError(t, err)
	require.False(t, result.Passed)
	require.False(t, result.Checks.SignalErrors)
}
