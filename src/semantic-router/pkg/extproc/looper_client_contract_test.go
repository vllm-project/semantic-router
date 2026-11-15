package extproc

import (
	"encoding/json"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// looperJSONSchemaFormat is a client's structured-output contract: a named,
// strict JSON schema.
const looperJSONSchemaFormat = `{"type":"json_schema","json_schema":{"name":"capital","strict":true,` +
	`"schema":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false}}}`

// TestLooperStagesKeepClientJSONSchema holds the Looper algorithms that
// rebuild the request around a stage prompt (the Fusion judge, the Flow
// final, the ReMoM synthesis) to the client's response_format, schema
// included, on every call they make, and to the answer of their last stage.
func TestLooperStagesKeepClientJSONSchema(t *testing.T) {
	request := `{"model":"vllm-sr/auto","response_format":` + looperJSONSchemaFormat +
		`,"messages":[{"role":"user","content":"What is the capital of France?"}]}`
	cases := map[string]struct {
		routes []string
		answer string
	}{
		"fusion-one-call":  {routes: []string{"judge", "model-a", "model-b", "model-c"}, answer: `{"selected_model":"model-b","rationale":"scripted"}`},
		"workflows-static": {routes: []string{"judge", "model-a", "model-b"}, answer: `{"selected_model":"model-b","rationale":"scripted"}`},
		"remom":            {routes: []string{"model-a", "model-a", "model-b", "model-b"}, answer: "model-b says Paris in detail (4df3f5a7)"},
	}
	for _, fixture := range looperFixtures {
		want, ok := cases[fixture.name]
		if !ok {
			continue
		}
		harness := newLooperHarness(t, fixture)
		for adapter, engine := range looperAdapters {
			t.Run(fixture.name+"/"+adapter, func(t *testing.T) {
				record := harness.serve(t, engine(harness.router), request)

				require.Equal(t, 200, record.Response.Status, "client reply: %s", record.Response.Body)
				routes := []string{}
				for _, call := range record.Calls {
					routes = append(routes, call.Route)
					var body struct {
						ResponseFormat json.RawMessage `json:"response_format"`
					}
					require.NoError(t, json.Unmarshal(call.Body, &body))
					assert.JSONEq(t, looperJSONSchemaFormat, string(body.ResponseFormat), "the call to %s", call.Route)
				}
				assert.Equal(t, want.routes, routes)
				var reply struct {
					Choices []struct {
						Message struct {
							Content string `json:"content"`
						} `json:"message"`
					} `json:"choices"`
				}
				require.NoError(t, json.Unmarshal(record.Response.Body, &reply))
				require.Len(t, reply.Choices, 1)
				assert.Equal(t, want.answer, reply.Choices[0].Message.Content)
			})
		}
	}
}

// TestLooperFailureKeepsUpstreamStatus answers a Looper run that a model's
// rate limit or outage ended with that status, as the client would get it
// from the model directly, rather than with the Router's own 500: a Flow
// worker that fails, a ReMoM round, buffered or streamed, whose every call
// fails under on_error: skip, and a Fusion panel that ends below quorum.
func TestLooperFailureKeepsUpstreamStatus(t *testing.T) {
	fixtures := map[string]looperFixture{}
	for _, fixture := range looperFixtures {
		fixtures[fixture.name] = fixture
	}
	codes := map[int]string{429: "rate_limited", 503: "upstream_unavailable"}
	cases := []struct {
		fixture  string
		failing  []string
		upstream int
		request  string
	}{
		{fixture: "workflows-static", failing: []string{"model-a"}, upstream: 429, request: looperClientRequest},
		{fixture: "workflows-static", failing: []string{"model-a"}, upstream: 503, request: looperClientRequest},
		{fixture: "remom", failing: []string{"model-a", "model-b"}, upstream: 429, request: looperClientRequest},
		{fixture: "remom", failing: []string{"model-a", "model-b"}, upstream: 429, request: looperClientStreamed},
		{fixture: "fusion-one-call", failing: []string{"model-a", "model-b", "model-c"}, upstream: 503, request: looperClientRequest},
	}
	for _, tc := range cases {
		fixture := fixtures[tc.fixture]
		fixture.failing, fixture.failStatus = tc.failing, tc.upstream
		harness := newLooperHarness(t, fixture)
		for adapter, engine := range looperAdapters {
			name := fmt.Sprintf("%s/%d/streamed=%t/%s", tc.fixture, tc.upstream, tc.request == looperClientStreamed, adapter)
			t.Run(name, func(t *testing.T) {
				record := harness.serve(t, engine(harness.router), tc.request)

				assert.Equal(t, tc.upstream, record.Response.Status, "client reply: %s", record.Response.Body)
				var reply struct {
					Error struct {
						Code string `json:"code"`
					} `json:"error"`
				}
				require.NoError(t, json.Unmarshal(record.Response.Body, &reply))
				assert.Equal(t, codes[tc.upstream], reply.Error.Code)
			})
		}
	}
}
