//go:build !windows

package apiserver

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

func TestLocalSystemOneAnswerGivesPoolRefusalsTheWorkerCodes(t *testing.T) {
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
		result     modelservice.SystemOneResult
		err        error
		wantStatus int
		wantCode   string
	}{
		{"every worker full", modelservice.SystemOneResult{}, fmt.Errorf("pool: %w", modelservice.ErrOverloaded), http.StatusTooManyRequests, "overloaded"},
		{"no ready worker", modelservice.SystemOneResult{}, modelservice.ErrUnavailable, http.StatusServiceUnavailable, "not_ready"},
		{"worker refused", modelservice.SystemOneResult{Status: 429, Body: []byte(`{"error":{"code":"overloaded","message":"queue full"}}`)}, modelservice.ErrOverloaded, 429, "overloaded"},
		{"answered", modelservice.SystemOneResult{Status: 200, Body: []byte(`{"answers":{}}`)}, nil, 200, ""},
		{"transport failure", modelservice.SystemOneResult{}, modelservice.ErrFailed, 0, ""},
	} {
		t.Run(test.name, func(t *testing.T) {
			status, body, err := localSystemOneAnswer(test.result, test.err)
			if status != test.wantStatus || code(body) != test.wantCode || !errors.Is(err, test.err) {
				t.Fatalf("got %d %s %v", status, body, err)
			}
		})
	}
}
