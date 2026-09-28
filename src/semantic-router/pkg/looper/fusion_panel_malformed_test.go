/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package looper

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestFusionLooperMalformedHTTPSuccessDoesNotMeetQuorum(t *testing.T) {
	var judgeCalls atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload struct {
			Model string `json:"model"`
		}
		require.NoError(t, json.NewDecoder(r.Body).Decode(&payload))
		switch payload.Model {
		case "panel-a":
			writeFusionTestCompletion(w, payload.Model, "panel a answer", http.StatusOK)
		case "panel-b":
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"choices":[`))
		case "panel-c":
			writeFusionTestCompletion(w, payload.Model, "failed", http.StatusBadGateway)
		case "judge":
			judgeCalls.Add(1)
			writeFusionTestCompletion(w, payload.Model, "unexpected judge response", http.StatusOK)
		default:
			t.Errorf("unexpected model call: %s", payload.Model)
		}
	}))
	defer server.Close()

	req := newFusionTestRequest()
	req.Algorithm = &config.AlgorithmConfig{
		Type: "fusion",
		Fusion: &config.FusionAlgorithmConfig{
			Model:                  "judge",
			AnalysisModels:         []string{"panel-a", "panel-b", "panel-c"},
			MaxConcurrent:          3,
			MinSuccessfulResponses: 2,
			OnError:                config.FusionOnErrorSkip,
		},
	}

	_, err := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(context.Background(), req)
	require.Error(t, err)
	assert.Equal(t, "fusion panel quorum not met: got 1 usable response, require 2", err.Error())
	assert.Zero(t, judgeCalls.Load(), "judge must not run below panel quorum")

	evidence, ok := FusionQuorumEvidenceFromError(err)
	require.True(t, ok)
	assert.Equal(t, 2, evidence.RequiredCount)
	assert.Equal(t, 1, evidence.UsableCount)
	assert.Equal(t, fusionTestTokenUsage("panel-a"), evidence.Usage)
	require.Len(t, evidence.Attempts, 3)
	assert.Equal(t, []FusionPanelAttemptState{
		FusionPanelAttemptUsable,
		FusionPanelAttemptFailed,
		FusionPanelAttemptFailed,
	}, []FusionPanelAttemptState{
		evidence.Attempts[0].State,
		evidence.Attempts[1].State,
		evidence.Attempts[2].State,
	})
	assert.Contains(t, evidence.Attempts[1].Error, "failed to parse response")
	assert.Contains(t, evidence.Attempts[2].Error, "request failed with status 502")
}
