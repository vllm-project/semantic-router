package recipe

import (
	"context"
	"encoding/json"
	"math"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

const rawValueProbeManifest = `schema_version: v1
name: raw-observation
routing_assets: {yaml: config.yaml, dsl: recipe.dsl}
coverage: {}
decisions:
  - id: example
    expected_decision: route
    expected_algorithm: static
    expected_signal_values:
      embedding:information: {gte: -1, lte: 1}
    variants:
      - id: inherited
        query: Describe a neutral item.
      - id: replaced
        query: Count two neutral items.
        expected_signal_values:
          structure:count: {gte: 2}
      - id: cleared
        query: Describe another neutral item.
        expected_signal_values: {}
`

func TestRawSignalValueDecodeAndMaterialization(t *testing.T) {
	manifest, err := decodeProbes([]byte(rawValueProbeManifest))
	require.NoError(t, err)
	probes, _ := flattenProbes(manifest)
	require.Len(t, probes, 3)
	require.Len(t, probes[0].Expected.SignalValues, 1)
	require.Equal(t, -1.0, *probes[0].Expected.SignalValues["embedding:information"].GTE)
	require.Equal(t, 2.0, *probes[1].Expected.SignalValues["structure:count"].GTE)
	require.Empty(t, probes[2].Expected.SignalValues)
	cloned := cloneExpectedAssertions(probes[0].Expected)
	*cloned.SignalValues["embedding:information"].GTE = 0
	require.Equal(t, -1.0, *probes[0].Expected.SignalValues["embedding:information"].GTE)
	for _, probe := range probes {
		request, materializeErr := materializeEvalRequest(probe)
		require.NoError(t, materializeErr)
		encoded, marshalErr := json.Marshal(request)
		require.NoError(t, marshalErr)
		require.NotContains(t, string(encoded), "signal_values")
	}
}

func TestRawSignalValueRejectsMalformedBounds(t *testing.T) {
	for _, value := range []string{
		"{}", "[]", "null", "{gt: 0}", "{gte: true}", "{gte: '0'}",
		"{gte: null, lte: 1}", "{gte: .nan}", "{lte: .inf}", "{gte: 2, lte: 1}", "{gte: 0, gte: 1}",
	} {
		t.Run(value, func(t *testing.T) {
			_, err := decodeProbes([]byte(strings.Replace(rawValueProbeManifest, "{gte: -1, lte: 1}", value, 1)))
			require.ErrorContains(t, err, "expected_signal_values")
		})
	}
	for _, value := range []string{"null", "[]", "{bad: {gte: 0}}"} {
		_, err := decodeProbes([]byte(strings.Replace(rawValueProbeManifest, "expected_signal_values: {}", "expected_signal_values: "+value, 1)))
		require.ErrorContains(t, err, "expected_signal_values")
	}
}

func rawValueResponse(value any) map[string]any {
	return map[string]any{
		"recipe": "default", "routing_decision": "route", "selected_model": "worker",
		"selection_status": "selected", "selection_method": "static", "recommended_models": []string{"worker"},
		"signal_values":   map[string]any{"embedding:information": value},
		"decision_result": map[string]any{"algorithm": "static"},
		"eval_trace":      []map[string]any{{"decision_name": "route", "matched": true}},
	}
}

func TestRawSignalValueActualPreviewComparison(t *testing.T) {
	manifest, err := decodeProbes([]byte(rawValueProbeManifest))
	require.NoError(t, err)
	probes, _ := flattenProbes(manifest)
	probe := probes[0]
	for _, test := range []struct {
		name  string
		value any
		want  bool
	}{
		{"below match threshold", .3, true},
		{"lower boundary", -1.0, true},
		{"upper boundary", 1.0, true},
		{"below range", -2, false},
		{"above range", 2, false},
		{"bool", true, false},
		{"null", nil, false},
		{"string", "0.3", false},
	} {
		t.Run(test.name, func(t *testing.T) {
			raw, marshalErr := json.Marshal(rawValueResponse(test.value))
			require.NoError(t, marshalErr)
			actual, checks, failures, compareErr := compareEvalResponse(raw, probe, probes)
			require.NoError(t, compareErr)
			require.Equal(t, test.want, checks.SignalValues, failures)
			require.Equal(t, test.want, len(failures) == 0)
			require.Contains(t, actual.SignalValues, "embedding:information")
		})
	}
	for _, reason := range []string{"missing", "corresponding error", "nested only", "required match"} {
		t.Run(reason, func(t *testing.T) {
			response := rawValueResponse(.3)
			current := probe
			switch reason {
			case "missing":
				delete(response, "signal_values")
			case "nested only":
				response["decision_result"].(map[string]any)["signal_values"] = response["signal_values"]
				delete(response, "signal_values")
			case "corresponding error":
				response["signal_errors"] = map[string]string{"embedding:information": "failed"}
			case "required match":
				current.Expected.Signals = map[string][]string{"embeddings": {"information"}}
			}
			raw, marshalErr := json.Marshal(response)
			require.NoError(t, marshalErr)
			_, _, failures, compareErr := compareEvalResponse(raw, current, probes)
			require.NoError(t, compareErr)
			require.NotEmpty(t, failures)
		})
	}
	passed, failures := compareSignalValues(probe.Expected.SignalValues, map[string]any{"embedding:information": math.NaN()}, nil)
	require.False(t, passed)
	require.NotEmpty(t, failures)
}

func TestRawSignalValuesUseCanonicalPreviewWire(t *testing.T) {
	manifest, err := decodeProbes([]byte(rawValueProbeManifest))
	require.NoError(t, err)
	probes, _ := flattenProbes(manifest)
	// EvalResponse serializes raw scores at the top level; decision_result
	// contains decision metadata and matched signals, not raw observations.
	raw := json.RawMessage(`{
		"recipe":"default","routing_decision":"route","selected_model":"worker",
		"selection_status":"selected","selection_method":"static",
		"recommended_models":["worker"],"signal_values":{"embedding:information":0.3},
		"decision_result":{"decision_name":"route","algorithm":"static","matched_signals":{}},
		"eval_trace":[{"decision_name":"route","matched":true}]
	}`)
	actual, checks, failures, err := compareEvalResponse(raw, probes[0], probes)
	require.NoError(t, err)
	require.Empty(t, failures)
	require.True(t, checks.SignalValues)
	require.Equal(t, .3, actual.SignalValues["embedding:information"])

	for _, test := range []struct {
		top, nested float64
		want        bool
	}{
		{.3, 2, true},
		{2, .3, false},
	} {
		wire := rawValueResponse(test.top)
		wire["decision_result"].(map[string]any)["signal_values"] = map[string]any{"embedding:information": test.nested}
		raw, err = json.Marshal(wire)
		require.NoError(t, err)
		actual, checks, _, err = compareEvalResponse(raw, probes[0], probes)
		require.NoError(t, err)
		require.Equal(t, test.want, checks.SignalValues)
		require.Equal(t, test.top, actual.SignalValues["embedding:information"])
	}
}

func TestServiceValidationCarriesAndEnforcesRawSignalValues(t *testing.T) {
	directory := writeManagedRecipe(t)
	writeFile(t, directory, "probes.yaml", strings.Replace(rawValueProbeManifest, "name: raw-observation", "name: test-recipe", 1))
	value := any(.3)
	service := NewService(Options{Directory: directory, Evaluator: evaluatorFunc(func(_ context.Context, request EvalRequest) (json.RawMessage, error) {
		encoded, err := json.Marshal(request)
		require.NoError(t, err)
		require.NotContains(t, string(encoded), "signal_values")
		response := rawValueResponse(value)
		response["requested_model"] = request.Model
		return json.Marshal(response)
	})})
	result, err := service.Validate(context.Background(), "example", "inherited")
	require.NoError(t, err)
	require.True(t, result.Passed, result.Failures)
	require.True(t, result.Checks.SignalValues)
	require.Equal(t, .3, result.Actual.SignalValues["embedding:information"])
	encoded, err := json.Marshal(result)
	require.NoError(t, err)
	require.Contains(t, string(encoded), `"signal_values":{"embedding:information":{"gte":-1,"lte":1}}`)
	value = "0.3"
	result, err = service.Validate(context.Background(), "example", "inherited")
	require.NoError(t, err)
	require.False(t, result.Passed)
	require.False(t, result.Checks.SignalValues)
}

func TestComplexityRawErrorsStayWithCompleteRuleName(t *testing.T) {
	for _, test := range []struct {
		key, failure string
		related      bool
	}{
		{"complexity:depth", "complexity:other:hard", false},
		{"complexity:depth", "complexity:depth:hard", true},
		{"complexity:depth:margin", "complexity:other:hard", false},
		{"complexity:literal:part:margin", "complexity:literal:other:hard", false},
		{"complexity:literal:part:margin", "complexity:literal:part:hard", true},
		{"complexity:literal:hard:margin", "complexity:literal:hard", false},
		{"complexity:literal:hard:margin", "complexity:literal:hard:easy", true},
	} {
		t.Run(test.key+"/"+test.failure, func(t *testing.T) {
			lower := -1.0
			passed, failures := compareSignalValues(
				map[string]SignalValueBounds{test.key: {GTE: &lower}},
				map[string]any{test.key: 0.0}, map[string]string{test.failure: "failed"})
			require.Equal(t, !test.related, passed, failures)
		})
	}
}
