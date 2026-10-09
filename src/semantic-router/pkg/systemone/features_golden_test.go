package systemone

import (
	"bytes"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestPolicyFeaturesMatchResearchCollector(t *testing.T) {
	path := filepath.Join("..", "..", "..", "..", "tools", "calibration", "systemone_auto", "tests", "fixtures", "native-features.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		FeatureNames []string `json:"feature_names"`
		Cases        []struct {
			Name     string          `json:"name"`
			Request  json.RawMessage `json:"request"`
			Response json.RawMessage `json:"response"`
			Features []float64       `json:"features"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(fixture.FeatureNames, FeatureNames) {
		t.Fatal("feature contract differs")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			var compact bytes.Buffer
			if err := json.Compact(&compact, tc.Request); err != nil {
				t.Fatal(err)
			}
			request, err := ParseNativeRequest(compact.Bytes())
			if err != nil {
				t.Fatal(err)
			}
			features := request.Features(request.Observe(tc.Response))
			if len(features) != len(tc.Features) {
				t.Fatal("feature lengths differ")
			}
			for i, value := range features {
				if math.Abs(value-tc.Features[i]) > 1e-12 {
					t.Errorf("%s: Go=%g research=%g", FeatureNames[i], value, tc.Features[i])
				}
			}
		})
	}
}
