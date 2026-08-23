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
	"encoding/json"
	"testing"
)

func TestParseStreamingResponseUsesNeutralTerminalUsage(t *testing.T) {
	body := []byte("data: {\"id\":\"chatcmpl-1\",\"model\":\"model-a\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"}}]}\n\n" +
		"data: {\"id\":\"chatcmpl-1\",\"model\":\"model-a\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\n" +
		"data: {\"id\":\"chatcmpl-1\",\"model\":\"model-a\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n" +
		"data: {\"id\":\"chatcmpl-1\",\"model\":\"model-a\",\"choices\":[],\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":8,\"total_tokens\":20}}\n\n" +
		"data: [DONE]\n\n")

	response, err := (&Client{}).parseStreamingResponse(body, "model-a")
	if err != nil {
		t.Fatalf("parseStreamingResponse() error = %v", err)
	}
	if response.Content != "hi" {
		t.Fatalf("content = %q, want hi", response.Content)
	}
	want := TokenUsage{PromptTokens: 12, CompletionTokens: 8, TotalTokens: 20}
	if response.Usage != want {
		t.Fatalf("usage = %+v, want %+v", response.Usage, want)
	}
}

func TestParseStreamingResponseAcceptsSSEDataWithoutSpace(t *testing.T) {
	body := []byte("data:{\"id\":\"chatcmpl-1\",\"model\":\"model-a\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\r\n\r\n" +
		"data:{\"id\":\"chatcmpl-1\",\"model\":\"model-a\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\r\n\r\n" +
		"data:{\"id\":\"chatcmpl-1\",\"model\":\"model-a\",\"choices\":[],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":4,\"total_tokens\":11}}\r\n\r\n" +
		"data:[DONE]\r\n\r\n")

	response, err := (&Client{}).parseStreamingResponse(body, "model-a")
	if err != nil {
		t.Fatalf("parseStreamingResponse() error = %v", err)
	}
	if response.Content != "hi" || response.Usage.TotalTokens != 11 {
		t.Fatalf("response = %+v", response)
	}
}

func TestParseStreamingResponseMissingUsageIsUnknown(t *testing.T) {
	body := []byte("data: {\"id\":\"chatcmpl-1\",\"model\":\"model-a\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\n" +
		"data: {\"id\":\"chatcmpl-1\",\"model\":\"model-a\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n" +
		"data: [DONE]\n\n")

	response, err := (&Client{}).parseStreamingResponse(body, "model-a")
	if err != nil {
		t.Fatalf("parseStreamingResponse() error = %v", err)
	}
	if response.Usage.Authoritative() {
		t.Fatalf("missing terminal usage became authoritative: %+v", response.Usage)
	}
}

func TestParseStreamingResponseRejectsInvalidTerminalUsage(t *testing.T) {
	body := []byte("data: {\"id\":\"chatcmpl-1\",\"model\":\"model-a\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\n" +
		"data: {\"id\":\"chatcmpl-1\",\"model\":\"model-a\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n" +
		"data: {\"id\":\"chatcmpl-1\",\"model\":\"model-a\",\"choices\":[],\"usage\":{\"prompt_tokens\":9,\"completion_tokens\":4,\"total_tokens\":-13}}\n\n" +
		"data: [DONE]\n\n")

	if _, err := (&Client{}).parseStreamingResponse(body, "model-a"); err == nil {
		t.Fatal("invalid terminal usage was accepted")
	}
}

func TestSetStreamParamStreamingRequestsIncludeUsage(t *testing.T) {
	out, err := setStreamParam([]byte(`{"model":"x"}`), true)
	if err != nil {
		t.Fatalf("setStreamParam() error = %v", err)
	}
	var request map[string]interface{}
	if err := json.Unmarshal(out, &request); err != nil {
		t.Fatalf("decode request: %v", err)
	}
	options, ok := request["stream_options"].(map[string]interface{})
	if !ok || options["include_usage"] != true {
		t.Fatalf("stream_options = %#v", request["stream_options"])
	}
}

func TestSetStreamParamNonStreamingDropsStreamOptions(t *testing.T) {
	out, err := setStreamParam([]byte(`{"model":"x","stream_options":{"include_usage":true}}`), false)
	if err != nil {
		t.Fatalf("setStreamParam() error = %v", err)
	}
	var request map[string]interface{}
	if err := json.Unmarshal(out, &request); err != nil {
		t.Fatalf("decode request: %v", err)
	}
	if _, found := request["stream_options"]; found {
		t.Fatalf("stream_options were retained: %s", out)
	}
}
