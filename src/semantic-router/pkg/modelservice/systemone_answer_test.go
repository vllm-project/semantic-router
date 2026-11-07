//go:build !windows

package modelservice

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"testing"
)

func TestSystemOneAnswerGivesPoolRefusalsTheWorkerCodes(t *testing.T) {
	code := func(body []byte) string {
		var answer struct {
			Error struct{ Code string } `json:"error"`
		}
		if json.Unmarshal(body, &answer) != nil {
			return ""
		}
		return answer.Error.Code
	}
	for _, test := range []struct {
		name       string
		result     SystemOneResult
		err        error
		wantStatus int
		wantCode   string
	}{
		{"every worker full", SystemOneResult{}, fmt.Errorf("pool: %w", ErrOverloaded), http.StatusTooManyRequests, "overloaded"},
		{"no ready worker", SystemOneResult{}, ErrUnavailable, http.StatusServiceUnavailable, "not_ready"},
		{"worker refused", SystemOneResult{Status: 429, Body: []byte(`{"error":{"code":"overloaded","message":"queue full"}}`)}, ErrOverloaded, 429, "overloaded"},
		{"answered", SystemOneResult{Status: 200, Body: []byte(`{"answers":{}}`)}, nil, 200, ""},
		{"transport failure", SystemOneResult{}, ErrFailed, 0, ""},
	} {
		t.Run(test.name, func(t *testing.T) {
			status, body, err := SystemOneAnswer(test.result, test.err)
			if status != test.wantStatus || code(body) != test.wantCode || !errors.Is(err, test.err) {
				t.Fatalf("got %d %s %v", status, body, err)
			}
		})
	}
}
