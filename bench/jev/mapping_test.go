package main

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestOrderedDistributionIgnoresResponseOrder(t *testing.T) {
	inputs := []string{
		`{"choice":"coding","confidence":0.6,"probabilities":{"writing":0.2,"other":0.1,"coding":0.7}}`,
		`{"choice":"coding","confidence":0.6,"probabilities":{"coding":0.7,"writing":0.2,"other":0.1}}`,
	}
	for _, raw := range inputs {
		var a answer
		if err := json.Unmarshal([]byte(raw), &a); err != nil {
			t.Fatal(err)
		}
		for _, scenario := range []struct {
			order []string
			want  []float32
		}{
			{[]string{"coding", "writing", "other"}, []float32{0.7, 0.2, 0.1}},
			{[]string{"other", "coding", "writing"}, []float32{0.1, 0.7, 0.2}},
		} {
			got, err := orderedDistribution(a, scenario.order)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got.Probabilities, scenario.want) {
				t.Fatalf("order %v: got %v, want %v", scenario.order, got.Probabilities, scenario.want)
			}
		}
		if *a.Probabilities["coding"] != 0.7 || *a.Confidence != 0.6 {
			t.Fatal("conversion changed source probabilities or confidence")
		}
	}
}

func TestOrderedDistributionRejectsInvalidMapping(t *testing.T) {
	var a answer
	if err := json.Unmarshal([]byte(`{"probabilities":{"coding":0.7,"writing":0.2,"other":0.1}}`), &a); err != nil {
		t.Fatal(err)
	}
	for _, order := range [][]string{
		nil,
		{"coding", "writing"},
		{"coding", "writing", "unknown"},
		{"coding", "coding", "other"},
	} {
		if _, err := orderedDistribution(a, order); err == nil {
			t.Fatalf("accepted invalid order %v", order)
		}
	}
}
