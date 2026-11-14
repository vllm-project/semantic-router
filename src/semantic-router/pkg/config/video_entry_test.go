package config

import (
	"strings"
	"testing"
)

func TestVideoAPIFormatMapsToItsCatalogProtocol(t *testing.T) {
	if got := CanonicalAPIFormat(APIFormatVideos); got != "videos" {
		t.Fatalf("canonical videos API format = %q", got)
	}
	if got := catalogProtocolForAPIFormat(APIFormatVideos); got != "openai/videos@1" {
		t.Fatalf("videos catalog protocol = %q", got)
	}
	format, err := apiFormatForProtocol("openai/videos@1")
	if err != nil || format != APIFormatVideos {
		t.Fatalf("videos runtime API format = %q, %v", format, err)
	}
}

func TestVideosRouteModelRejectsColon(t *testing.T) {
	_, err := ParseYAMLBytes([]byte(`version: v0.5
providers:
  defaults: {model: video:route}
  models:
    - name: video:route
      api_format: videos
      capabilities: [video_generation]
      backend_refs: [{endpoint: "127.0.0.1:8000", protocol: http, provider: vllm}]
recipes:
  - name: default
`))
	if err == nil || !strings.Contains(err.Error(), "cannot contain ':'") {
		t.Fatalf("colon videos route error = %v", err)
	}
}

func TestVideoGenerationWarnsWhenRouteHasSeveralEndpoints(t *testing.T) {
	cfg := &RouterConfig{BackendModels: BackendModels{
		ModelConfig: map[string]ModelParams{
			"video-route": {PreferredEndpoints: []string{"one", "two"}, Capabilities: []string{"video_generation"}},
		},
		VLLMEndpoints: []VLLMEndpoint{{Name: "one"}, {Name: "two"}},
	}}
	warnings := videoGenerationWarnings(cfg)
	if len(warnings) != 1 || warnings[0].Code != "video_generation_multiple_endpoints" {
		t.Fatalf("video warnings = %+v", warnings)
	}
}
