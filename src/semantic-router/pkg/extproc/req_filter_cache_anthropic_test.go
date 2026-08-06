package extproc

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	httputil "github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/http"
)

func TestTranslateAnthropicCacheHitRestoresCacheUsage(t *testing.T) {
	cached := []byte(`{
		"id":"cached",
		"model":"claude",
		"choices":[{
			"index":0,
			"message":{"role":"assistant","content":"cached"},
			"finish_reason":"stop"
		}],
		"usage":{
			"prompt_tokens":100,
			"completion_tokens":10,
			"total_tokens":110,
			"prompt_tokens_details":{
				"cached_tokens":30,
				"cache_write_tokens":20
			}
		}
	}`)
	ctx := &RequestContext{
		ClientProtocol: config.ClientProtocolAnthropic,
		RequestModel:   "claude",
	}
	hydrateAnthropicCacheUsage(ctx, cached)
	response := httputil.CreateCacheHitResponse(cached, false, "", "", nil, 1)

	translated := translateAnthropicCacheHit(ctx, response)
	var body map[string]interface{}
	if err := json.Unmarshal(translated.GetImmediateResponse().GetBody(), &body); err != nil {
		t.Fatalf("decode Anthropic cache hit: %v", err)
	}
	usage := body["usage"].(map[string]interface{})
	if usage["input_tokens"] != float64(50) ||
		usage["cache_read_input_tokens"] != float64(30) ||
		usage["cache_creation_input_tokens"] != float64(20) {
		t.Fatalf("unexpected Anthropic cache usage: %#v", usage)
	}
}

func TestTranslateAnthropicCacheHitStreamingWireFormat(t *testing.T) {
	cached := []byte(`{
		"id":"cached",
		"model":"claude",
		"choices":[{
			"index":0,
			"message":{"role":"assistant","content":"cached"},
			"finish_reason":"stop"
		}],
		"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}
	}`)
	ctx := &RequestContext{
		ClientProtocol:          config.ClientProtocolAnthropic,
		ExpectStreamingResponse: true,
		RequestModel:            "claude",
	}
	response := httputil.CreateCacheHitResponse(cached, true, "", "", nil, 1)

	translated := translateAnthropicCacheHit(ctx, response)
	body := string(translated.GetImmediateResponse().GetBody())

	for _, expected := range []string{
		"event: message_start",
		"event: content_block_delta",
		"event: message_stop",
	} {
		if !strings.Contains(body, expected) {
			t.Fatalf("Anthropic cache stream missing %q:\n%s", expected, body)
		}
	}
}
