package extproc

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

const mistralProviderModelID = "mistralai/Mistral-Small-4-119B-2603"

func TestTopLevelReasoningEffortFamilyForLocalVLLM(t *testing.T) {
	router := newTopLevelReasoningEffortRouter()
	decision := router.Config.GetDecisionByName("mistral-analysis")

	t.Run("enabled writes only the top-level effort", func(t *testing.T) {
		requestBody := []byte(`{
			"model":"local/mistral-small-4",
			"messages":[{"role":"user","content":"analyze"}],
			"chat_template_kwargs":{"reasoning_effort":"medium","unrelated":"keep"}
		}`)

		modified, err := router.setReasoningModeToRequestBodyForProvider(
			requestBody,
			true,
			decision,
			nil,
		)
		require.NoError(t, err)

		request := unmarshalReasoningRequest(t, modified)
		assert.Equal(t, "high", request["reasoning_effort"])
		assert.Equal(t, map[string]interface{}{"unrelated": "keep"}, request["chat_template_kwargs"])
	})

	t.Run("disabled does not synthesize an effort", func(t *testing.T) {
		requestBody := []byte(`{
			"model":"local/mistral-small-4",
			"messages":[{"role":"user","content":"answer"}]
		}`)

		modified, err := router.setReasoningModeToRequestBodyForProvider(
			requestBody,
			false,
			decision,
			nil,
		)
		require.NoError(t, err)

		request := unmarshalReasoningRequest(t, modified)
		assertReasoningEffortAbsent(t, request)
		assertChatTemplateAbsent(t, request)
	})

	t.Run("disabled preserves a caller supplied top-level effort", func(t *testing.T) {
		requestBody := []byte(`{
			"model":"local/mistral-small-4",
			"messages":[{"role":"user","content":"answer"}],
			"reasoning_effort":"low"
		}`)

		modified, err := router.setReasoningModeToRequestBodyForProvider(
			requestBody,
			false,
			decision,
			nil,
		)
		require.NoError(t, err)

		request := unmarshalReasoningRequest(t, modified)
		assert.Equal(t, "low", request["reasoning_effort"])
		assertChatTemplateAbsent(t, request)
	})
}

func TestBuildTopLevelReasoningEffortFieldsForLocalVLLM(t *testing.T) {
	router := newTopLevelReasoningEffortRouter()
	fields, effort := router.buildReasoningRequestFieldsForProvider(
		"local/mistral-small-4",
		true,
		router.Config.GetDecisionByName("mistral-analysis"),
		nil,
	)

	require.NotNil(t, fields)
	assert.Equal(t, "high", effort)
	assert.Equal(t, "high", fields["reasoning_effort"])
	_, hasTemplateKwargs := fields["chat_template_kwargs"]
	assert.False(t, hasTemplateKwargs)
}

func newTopLevelReasoningEffortRouter() *OpenAIRouter {
	return newReasoningRouter(
		config.ReasoningConfig{
			DefaultReasoningEffort: "medium",
			ReasoningFamilies: map[string]config.ReasoningFamilyConfig{
				"mistral": {
					Type:      config.ReasoningFamilyTypeTopLevelReasoningEffort,
					Parameter: "reasoning_effort",
				},
			},
		},
		[]config.Decision{
			reasoningDecision(
				"mistral-analysis",
				"",
				0,
				"local/mistral-small-4",
				boolPtr(true),
				"high",
			),
		},
		map[string]config.ModelParams{
			"local/mistral-small-4": {ReasoningFamily: "mistral"},
		},
	)
}

func TestTopLevelReasoningEffortPreservesOpaqueRequestFields(t *testing.T) {
	router := newTopLevelReasoningEffortRouter()
	requestBody := []byte(`{
		"model":"local/mistral-small-4",
		"messages":[{"role":"user","content":"analyze"}],
		"opaque_integer":9007199254740993
	}`)

	modified, err := router.setReasoningModeToRequestBody(
		requestBody,
		true,
		router.Config.GetDecisionByName("mistral-analysis"),
	)
	require.NoError(t, err)

	var request map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(modified, &request))
	assert.JSONEq(t, `9007199254740993`, string(request["opaque_integer"]))
}

func TestAutoRoutingAppliesExplicitTopLevelEffortThroughProviderModelAlias(t *testing.T) {
	router, decision := newTopLevelReasoningEffortFullPathRouter()
	requestBody := explicitTopLevelReasoningFullPathRequest()
	request, err := parseOpenAIRequest(requestBody)
	require.NoError(t, err)
	ctx := &RequestContext{
		OriginalRequestBody: requestBody,
		VSRSelectedDecision: decision,
	}

	modified, err := router.modifyRequestBodyForAutoRouting(
		request,
		mistralProviderModelID,
		decision.Name,
		true,
		nil,
		ctx,
	)
	require.NoError(t, err)

	assertExplicitTopLevelReasoningProviderBody(t, modified)
}

func TestLooperInternalPluginsApplyExplicitTopLevelEffortThroughProviderModelAlias(t *testing.T) {
	router, decision := newTopLevelReasoningEffortFullPathRouter()
	requestBody := explicitTopLevelReasoningFullPathRequest()
	ctx := &RequestContext{
		LooperRequest:           true,
		ExpectStreamingResponse: true,
		OriginalRequestBody:     requestBody,
		VSRSelectedDecision:     decision,
		Headers: map[string]string{
			headers.VSRLooperDecision: decision.Name,
		},
	}

	response, err := router.handleLooperInternalRequestWithPlugins(
		"local/mistral-small-4",
		ctx,
	)
	require.NoError(t, err)
	require.NotNil(t, response.GetRequestBody())

	modified := response.GetRequestBody().Response.GetBodyMutation().GetBody()
	assertExplicitTopLevelReasoningProviderBody(t, modified)
	assertLooperStreamingUsageFields(t, modified)
}

func newTopLevelReasoningEffortFullPathRouter() (*OpenAIRouter, *config.Decision) {
	decision := reasoningDecision(
		"mistral-analysis",
		"",
		100,
		"local/mistral-small-4",
		boolPtr(true),
		"high",
	)
	router := &OpenAIRouter{
		Cache: &spyCache{},
		Config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				ReasoningConfig: config.ReasoningConfig{
					DefaultReasoningEffort: "medium",
					ReasoningFamilies: map[string]config.ReasoningFamilyConfig{
						"mistral": {
							Type:      config.ReasoningFamilyTypeTopLevelReasoningEffort,
							Parameter: "reasoning_effort",
						},
					},
				},
				Decisions: []config.Decision{decision},
			},
			BackendModels: config.BackendModels{
				ModelConfig: map[string]config.ModelParams{
					"local/mistral-small-4": {
						PreferredEndpoints: []string{"mistral-backend"},
						ReasoningFamily:    "mistral",
						ExternalModelIDs: map[string]string{
							"vllm": mistralProviderModelID,
						},
					},
				},
				VLLMEndpoints: []config.VLLMEndpoint{
					{
						Name:    "mistral-backend",
						Address: "127.0.0.1",
						Port:    8000,
						Type:    "vllm",
						Weight:  1,
					},
				},
			},
		},
	}
	router.CredentialResolver = newTestCredentialResolver(router.Config)
	return router, &router.Config.Decisions[0]
}

func explicitTopLevelReasoningFullPathRequest() []byte {
	return []byte(`{
		"model":"vllm-sr/mom-v1-ultra",
		"messages":[{"role":"user","content":"analyze"}],
		"chat_template_kwargs":{"reasoning_effort":"medium","unrelated":"keep"},
		"opaque_integer":9007199254740993
	}`)
}

func assertExplicitTopLevelReasoningProviderBody(t *testing.T, body []byte) {
	t.Helper()
	var request map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(body, &request))
	assert.JSONEq(t, `"`+mistralProviderModelID+`"`, string(request["model"]))
	assert.JSONEq(t, `"high"`, string(request["reasoning_effort"]))
	assert.JSONEq(t, `9007199254740993`, string(request["opaque_integer"]))

	var kwargs map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(request["chat_template_kwargs"], &kwargs))
	_, hasNestedEffort := kwargs["reasoning_effort"]
	assert.False(t, hasNestedEffort)
	assert.JSONEq(t, `"keep"`, string(kwargs["unrelated"]))
}

func assertLooperStreamingUsageFields(t *testing.T, body []byte) {
	t.Helper()
	var request map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(body, &request))
	assert.JSONEq(t, `true`, string(request["stream"]))

	var streamOptions map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(request["stream_options"], &streamOptions))
	assert.JSONEq(t, `true`, string(streamOptions["include_usage"]))
}
