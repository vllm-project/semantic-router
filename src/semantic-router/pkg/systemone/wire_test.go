package systemone

import (
	"bytes"
	"compress/gzip"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/klauspost/compress/zstd"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func compressed(t *testing.T, encoding string, data []byte) []byte {
	t.Helper()
	var out bytes.Buffer
	switch encoding {
	case "gzip":
		writer := gzip.NewWriter(&out)
		_, _ = writer.Write(data)
		_ = writer.Close()
	case "zstd":
		writer, err := zstd.NewWriter(&out)
		if err != nil {
			t.Fatal(err)
		}
		_, _ = writer.Write(data)
		_ = writer.Close()
	}
	return out.Bytes()
}

func multipartBody(parts [][3]string) ([]byte, string) {
	const boundary = "b0undary"
	var out bytes.Buffer
	for _, part := range parts {
		fmt.Fprintf(&out, "--%s\r\nContent-Disposition: form-data; name=%q; filename=%q\r\n", boundary, part[0], part[0])
		if part[1] != "" {
			fmt.Fprintf(&out, "Content-Type: %s\r\n", part[1])
		}
		fmt.Fprintf(&out, "\r\n%s\r\n", part[2])
	}
	fmt.Fprintf(&out, "--%s--\r\n", boundary)
	return out.Bytes(), "multipart/form-data; boundary=" + boundary
}

type forwarded struct {
	bodies   []string
	response string
	status   int
}

func wireHandler(t *testing.T, sink *forwarded) http.HandlerFunc {
	t.Helper()
	cfg := &config.RouterConfig{}
	cfg.ModelDeployments = map[string]config.ModelDeployment{"private": {Provider: config.ModelRuntimeProvider, Artifact: "vllm-sr/test"}}
	listener := &config.Listener{SystemOne: &config.ListenerSystemOne{Models: []string{"vllm-sr/test", "vllm-sr/other"}}}
	return Handler(cfg, listener, func(_ context.Context, _ string, body json.RawMessage) (int, []byte, error) {
		sink.bodies = append(sink.bodies, string(body))
		status := sink.status
		if status == 0 {
			status = 200
		}
		response := sink.response
		if response == "" {
			response = `{"model":"private","answers":{"q":{"type":"noul","noul":0.5}},"usage":{"input_tokens":1,"output_tokens":0}}`
		}
		return status, []byte(response), nil
	})
}

func send(handler http.HandlerFunc, path string, body []byte, headers map[string]string) (int, map[string]string) {
	request := httptest.NewRequest(http.MethodPost, path, bytes.NewReader(body))
	for name, value := range headers {
		request.Header.Set(name, value)
	}
	response := httptest.NewRecorder()
	handler(response, request)
	var got struct {
		Error map[string]string `json:"error"`
	}
	_ = json.Unmarshal(response.Body.Bytes(), &got)
	return response.Code, got.Error
}

func TestPublicSystemOneReadsCompressedBodies(t *testing.T) {
	sink := &forwarded{}
	handler := wireHandler(t, sink)
	body := []byte(`{"model":"vllm-sr/test","state":"` + strings.Repeat("x", 3000) + `","questions":{"q":{"type":"noul","instructions":"?"}}}`)
	for _, encoding := range []string{"gzip", "x-gzip", "zstd"} {
		sink.bodies = nil
		wire := compressed(t, strings.TrimPrefix(encoding, "x-"), body)
		if status, _ := send(handler, "/v1/systemone", wire, map[string]string{"Content-Encoding": encoding}); status != 200 || len(sink.bodies) != 1 || sink.bodies[0] != string(body) {
			t.Fatalf("%s => %d, forwarded %d bodies", encoding, status, len(sink.bodies))
		}
	}
	for encoding, want := range map[string][2]string{"br": {"415", "encoding_unsupported"}, "gzip, gzip": {"415", "encoding_unsupported"}} {
		status, answer := send(handler, "/v1/systemone", body, map[string]string{"Content-Encoding": encoding})
		if fmt.Sprint(status) != want[0] || answer["reason"] != want[1] || answer["code"] != "unsupported_media_type" {
			t.Fatalf("%s => %d %v", encoding, status, answer)
		}
	}
	if status, answer := send(handler, "/v1/systemone", []byte("\x28\xb5\x2f\xfd"+strings.Repeat("\x00", 20)), map[string]string{"Content-Encoding": "zstd"}); status != 400 || answer["reason"] != "encoding_malformed" {
		t.Fatalf("corrupt zstd => %d %v", status, answer)
	}
	bomb := compressed(t, "gzip", make([]byte, RequestLimit+1))
	if status, _ := send(handler, "/v1/systemone", bomb, map[string]string{"Content-Encoding": "gzip"}); status != 413 {
		t.Fatalf("a body over the limit once decompressed => %d", status)
	}
}

func TestPublicSystemOneReadsMultipartAssets(t *testing.T) {
	sink := &forwarded{}
	handler := wireHandler(t, sink)
	image := "\x89PNG\r\n\x1a\nnot really"
	request := `{"model":"vllm-sr/test","state":{"photo":"<image:1>"},"questions":{"q":{"type":"noul","instructions":"red?"}},"assets":{"<image:1>":{"type":"image","part":"asset-0"}}}`
	body, contentType := multipartBody([][3]string{{"request", "application/json", request}, {"asset-0", "image/png", image}})
	for _, encoding := range []string{"", "zstd"} {
		sink.bodies = nil
		wire, headers := body, map[string]string{"Content-Type": contentType}
		if encoding != "" {
			wire, headers["Content-Encoding"] = compressed(t, encoding, body), encoding
		}
		status, answer := send(handler, "/v1/systemone", wire, headers)
		if status != 200 || len(sink.bodies) != 1 {
			t.Fatalf("multipart %q => %d %v", encoding, status, answer)
		}
		var canonical struct {
			State  json.RawMessage                       `json:"state"`
			Assets map[string]map[string]json.RawMessage `json:"assets"`
		}
		if json.Unmarshal([]byte(sink.bodies[0]), &canonical) != nil {
			t.Fatalf("forwarded %s", sink.bodies[0])
		}
		entry := canonical.Assets["<image:1>"]
		if string(canonical.State) != `{"photo":"<image:1>"}` || entry["part"] != nil || string(entry["data_base64"]) != `"`+base64.StdEncoding.EncodeToString([]byte(image))+`"` {
			t.Fatalf("canonical request %s", sink.bodies[0])
		}
	}
	for name, parts := range map[string][][3]string{
		"asset first":    {{"asset-0", "image/png", image}, {"request", "application/json", request}},
		"text part":      {{"request", "", request}, {"asset-0", "text/plain", image}},
		"unclaimed part": {{"request", "", request}, {"asset-0", "image/png", image}, {"asset-1", "", image}},
		"missing part":   {{"request", "", strings.Replace(request, "asset-0", "asset-9", 1)}, {"asset-0", "image/png", image}},
		"duplicate part": {{"request", "", request}, {"asset-0", "image/png", image}, {"asset-0", "image/png", image}},
	} {
		wire, contentType := multipartBody(parts)
		if status, answer := send(handler, "/v1/systemone", wire, map[string]string{"Content-Type": contentType}); status != 400 || answer["reason"] != "multipart_malformed" {
			t.Fatalf("%s => %d %v", name, status, answer)
		}
	}
}

func TestPublicSystemOneForwardsLongContext(t *testing.T) {
	sink := &forwarded{}
	handler := wireHandler(t, sink)
	for _, body := range []string{
		`{"model":"vllm-sr/test","options":{"long_context":true},"state":"s"}`,
		`{"model":"vllm-sr/test","options":{"deadline_ms":5,"long_context":false},"state":"s"}`,
	} {
		sink.bodies = nil
		if status, answer := send(handler, "/v1/systemone", []byte(body), nil); status != 200 || len(sink.bodies) != 1 || sink.bodies[0] != body {
			t.Fatalf("%s => %d %v, forwarded %q", body, status, answer, sink.bodies)
		}
	}
	if status, answer := send(handler, "/v1/systemone", []byte(`{"model":"vllm-sr/test","options":{"long_context":"yes"},"state":"s"}`), nil); status != 400 || !strings.Contains(answer["message"], "long_context") {
		t.Fatalf("a non-boolean long_context => %d %v", status, answer)
	}
}

func TestPublicSystemOneAnswersAFailedQuestionAsTheWholeRequest(t *testing.T) {
	sink := &forwarded{response: `{"model":"private","answers":{"ok":{"type":"noul","noul":0.5},"box":{"type":"box","error":"invalid_question","reason":"box_unsupported","message":"model /models/private does not answer box questions"}},"usage":{"input_tokens":1,"output_tokens":0}}`}
	handler := wireHandler(t, sink)
	body := []byte(`{"model":"vllm-sr/test","state":"s","questions":{"ok":{"type":"noul","instructions":"?"},"box":{"type":"box","instructions":"the mug"}}}`)
	status, answer := send(handler, "/v1/systemone", body, nil)
	if status != 422 || answer["code"] != "question_failed" || answer["question_error"] != "invalid_question" || answer["reason"] != "box_unsupported" || answer["question"] != "box" {
		t.Fatalf("/v1/systemone => %d %v", status, answer)
	}
	if strings.Contains(answer["message"], "/models/") {
		t.Fatalf("the runtime's message reached the client: %q", answer["message"])
	}
	if status, _ := send(handler, "/v1/decisions", body, nil); status != 200 {
		t.Fatalf("/v1/decisions must keep per-question errors, got %d", status)
	}
	sink.response = `{"model":"private","answers":{"q":{"type":"noul","noul":0.5}},"states":{"s2":{"model":"private","answers":{"r":{"type":"noul","error":"deadline_exceeded","reason":"deadline"}},"usage":{"input_tokens":1,"output_tokens":0}}},"usage":{"input_tokens":1,"output_tokens":0}}`
	if status, answer := send(handler, "/v1/systemone", body, nil); status != 504 || answer["question"] != "r" {
		t.Fatalf("a failed question of a states entry => %d %v", status, answer)
	}
	sink.status, sink.response = 400, `{"error":{"code":"invalid_request","message":"no question is valid (box: ...)","reason":"box_unsupported","question":"box"}}`
	if status, answer := send(handler, "/v1/systemone", body, nil); status != 422 || answer["reason"] != "box_unsupported" || answer["question"] != "box" {
		t.Fatalf("a request without a valid question on /v1/systemone => %d %v", status, answer)
	}
	if status, answer := send(handler, "/v1/decisions", body, nil); status != 400 || answer["code"] != "invalid_request" || !strings.Contains(answer["message"], "box questions") {
		t.Fatalf("a request without a valid question on /v1/decisions => %d %v", status, answer)
	}
}

func TestPublicSystemOneNamesTheModelsItServes(t *testing.T) {
	handler := wireHandler(t, &forwarded{})
	status, answer := send(handler, "/v1/systemone", []byte(`{"model":"unserved-model","state":"s","questions":{}}`), nil)
	if status != 403 || answer["code"] != "model_not_allowed" || !strings.Contains(answer["message"], "vllm-sr/test, vllm-sr/other") || !strings.Contains(answer["message"], `"unserved-model"`) {
		t.Fatalf("an unserved model => %d %v", status, answer)
	}
	response := httptest.NewRecorder()
	handler(response, httptest.NewRequest(http.MethodGet, "/v1/models", nil))
	var listing struct {
		Data   []map[string]any    `json:"data"`
		Models []map[string]string `json:"models"`
	}
	if response.Code != 200 || json.Unmarshal(response.Body.Bytes(), &listing) != nil || len(listing.Models) != len(listing.Data) {
		t.Fatalf("GET /v1/models => %d %s", response.Code, response.Body.String())
	}
	for _, model := range listing.Models {
		if model["name"] == "" || len(model) != 3 {
			t.Fatalf("model entry %v", model)
		}
	}
}
