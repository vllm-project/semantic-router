package modelservice

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestDecideRejectsAnswersThatDoNotFitTheirQuestion(t *testing.T) {
	kinds := []Choice{{Key: "code"}, {Key: "math"}}
	levels := []string{"easy", "medium", "hard"}
	cases := map[string]struct {
		question Question
		answer   map[string]interface{}
		valid    bool
	}{
		"choice against its distribution": {
			Question{Type: "choice", Choices: kinds},
			map[string]interface{}{"type": "choice", "choice": "code", "probabilities": map[string]float64{"code": 0.2, "math": 0.8}},
			false,
		},
		"undeclared option in place of a declared one": {
			Question{Type: "choice", Choices: kinds},
			map[string]interface{}{"type": "choice", "choice": "poetry", "probabilities": map[string]float64{"code": 0.1, "poetry": 0.9}},
			false,
		},
		"distribution summing to 1.6": {
			Question{Type: "choice", Choices: kinds},
			map[string]interface{}{"type": "choice", "choice": "math", "probabilities": map[string]float64{"code": 0.8, "math": 0.8}},
			false,
		},
		"choice answered as noul": {
			Question{Type: "choice", Choices: kinds},
			map[string]interface{}{"type": "noul", "noul": 0.9},
			false,
		},
		"noul above one": {Question{Type: "noul"}, map[string]interface{}{"type": "noul", "noul": 1.7}, false},
		"score past its levels": {
			Question{Type: "score", Levels: levels},
			map[string]interface{}{"type": "score", "score": 2.5, "probabilities": map[string]float64{"0": 0.1, "1": 0.3, "2": 0.6}},
			false,
		},
		"score over fewer levels": {
			Question{Type: "score", Levels: levels},
			map[string]interface{}{"type": "score", "score": 0.5, "probabilities": map[string]float64{"0": 0.5, "1": 0.5}},
			false,
		},
		"most probable choice": {
			Question{Type: "choice", Choices: kinds},
			map[string]interface{}{"type": "choice", "choice": "math", "probabilities": map[string]float64{"code": 0.2, "math": 0.8}},
			true,
		},
		"tied choice": {
			Question{Type: "choice", Choices: kinds},
			map[string]interface{}{"type": "choice", "choice": "code", "probabilities": map[string]float64{"code": 0.5, "math": 0.5}},
			true,
		},
		"no choice": {
			Question{Type: "choice", Choices: kinds},
			map[string]interface{}{"type": "choice", "probabilities": map[string]float64{"code": 0.5, "math": 0.5}},
			false,
		},
		"confidence above one": {
			Question{Type: "choice", Choices: kinds},
			map[string]interface{}{"type": "choice", "choice": "math", "confidence": 7.5, "probabilities": map[string]float64{"code": 0.2, "math": 0.8}},
			false,
		},
		"noul without a value": {Question{Type: "noul"}, map[string]interface{}{"type": "noul"}, false},
		"noul of null":         {Question{Type: "noul"}, map[string]interface{}{"type": "noul", "noul": nil}, false},
		"noul of zero":         {Question{Type: "noul"}, map[string]interface{}{"type": "noul", "noul": 0}, true},
		"score without a value": {
			Question{Type: "score", Levels: levels},
			map[string]interface{}{"type": "score", "probabilities": map[string]float64{"0": 1, "1": 0, "2": 0}},
			false,
		},
		"score against its distribution": {
			Question{Type: "score", Levels: levels},
			map[string]interface{}{"type": "score", "score": 2, "probabilities": map[string]float64{"0": 1, "1": 0, "2": 0}},
			false,
		},
		"score of zero": {
			Question{Type: "score", Levels: levels},
			map[string]interface{}{"type": "score", "score": 0, "probabilities": map[string]float64{"0": 1, "1": 0, "2": 0}},
			true,
		},
		"noul": {Question{Type: "noul"}, map[string]interface{}{"type": "noul", "noul": 0.4}, true},
		"score": {
			Question{Type: "score", Levels: levels},
			map[string]interface{}{"type": "score", "score": 1.5, "probabilities": map[string]float64{"0": 0.1, "1": 0.3, "2": 0.6}},
			true,
		},
		"runtime error": {Question{Type: "noul"}, map[string]interface{}{"type": "noul", "error": "max_length_exceeded"}, false},
	}
	request := Request{State: "text"}
	answers := map[string]interface{}{}
	for id, testCase := range cases {
		question := testCase.question
		question.ID, question.Instructions = id, "?"
		request.Questions = append(request.Questions, question)
		answers[id] = testCase.answer
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"model": "tiny", "answers": answers, "usage": map[string]int{"input_tokens": 1, "output_tokens": 0},
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
		switch id {
		case "runtime error":
			want = "max_length_exceeded"
		case "no choice", "noul without a value", "noul of null", "score without a value":
			want = "missing_answer_value"
		}
		if got.Error != want {
			t.Errorf("%s: error = %q, want %q (answer %+v)", id, got.Error, want, got)
		}
	}
}
