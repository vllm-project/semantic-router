package modelservice

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// A Set gives an independent probability to every declared label, so the
// probabilities need not sum to one. A map that leaves a declared label out must
// not reach the decision signal, which would read the omitted label as zero.
func TestDecideRejectsSetProbabilitiesThatMissDeclaredLabels(t *testing.T) {
	labels := []Choice{{Key: "billing"}, {Key: "shipping"}}
	cases := map[string]struct {
		probabilities map[string]float64
		valid         bool
	}{
		"every declared label":                  {map[string]float64{"billing": 0.9, "shipping": 0.8}, true},
		"zero for a declared label":             {map[string]float64{"billing": 0.9, "shipping": 0}, true},
		"declared label omitted":                {map[string]float64{"billing": 0.9}, false},
		"empty map":                             {map[string]float64{}, false},
		"undeclared label in place of declared": {map[string]float64{"billing": 0.9, "refunds": 0.8}, false},
		"undeclared extra label":                {map[string]float64{"billing": 0.9, "shipping": 0.8, "refunds": 0.1}, false},
	}
	request := Request{State: "text"}
	sets := map[string]interface{}{}
	for id, testCase := range cases {
		request.Questions = append(request.Questions, Question{ID: id, Type: "set", Instructions: "?", Labels: labels})
		sets[id] = map[string]interface{}{"selected": []string{}, "probabilities": testCase.probabilities}
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"model": "tiny", "answers": map[string]interface{}{}, "sets": sets,
			"usage": map[string]int{"input_tokens": 1, "output_tokens": 0},
		})
	}))
	defer server.Close()
	client, err := NewClient(server.URL)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	response, err := client.Decide(ctx, request)
	if err != nil {
		t.Fatal(err)
	}
	for id, testCase := range cases {
		got := response.Answers[id]
		want := ""
		if !testCase.valid {
			want = "invalid_model_output"
		}
		if got.Error != want {
			t.Errorf("%s: error = %q, want %q", id, got.Error, want)
		}
	}
}
