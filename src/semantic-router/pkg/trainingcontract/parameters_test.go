package trainingcontract

import (
	"encoding/json"
	"math"
	"reflect"
	"strconv"
	"testing"
)

func TestParameterNumbersRoundTrip(t *testing.T) {
	input := `{"parameters":{"epochs":100,"learning_rate":0.001,"nested":[256,{"weight":1.0,"scale":1e20,"enabled":true,"name":"model","optional":null}],"min":` + strconv.Itoa(math.MinInt) + `,"max":` + strconv.Itoa(math.MaxInt) + `}}`
	want := map[string]any{
		"epochs": 100, "learning_rate": 0.001,
		"nested": []any{256, map[string]any{"weight": float64(1), "scale": 1e20, "enabled": true, "name": "model", "optional": nil}},
		"min":    math.MinInt, "max": math.MaxInt,
	}
	var spec RunSpec
	if err := json.Unmarshal([]byte(input), &spec); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(map[string]any(spec.Parameters), want) {
		t.Fatalf("decoded parameters=%#v want=%#v", spec.Parameters, want)
	}
	encoded, err := json.Marshal(spec)
	if err != nil {
		t.Fatal(err)
	}
	var worker WorkerRequest
	// Both boundaries carry the same parameter object, including nested values.
	if operationErr := json.Unmarshal(encoded, &worker); operationErr != nil {
		t.Fatal(operationErr)
	}
	if !reflect.DeepEqual(map[string]any(worker.Parameters), want) {
		t.Fatalf("round trip changed parameters: %s", encoded)
	}
}

func TestParameterNumbersRejectOverflow(t *testing.T) {
	for _, value := range []string{"9223372036854775808", "-9223372036854775809", "1e309", "-1e309"} {
		t.Run(value, func(t *testing.T) {
			var spec RunSpec
			if err := json.Unmarshal([]byte(`{"parameters":{"nested":[{"value":`+value+`}]}}`), &spec); err == nil {
				t.Fatal("accepted out-of-range parameter")
			}
		})
	}
}
