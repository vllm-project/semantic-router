package routerreplay

import (
	"encoding/base64"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

// Replay is body-agnostic (it stores bytes), but the image vertical adds one
// requirement: a generated-image payload is a base64-encoded binary artifact,
// often larger than text bodies. This test pins that a real images-dialect
// request/response pair round-trips capture -> read -> restore byte-for-byte,
// including the artifact payloads, so a replayed image request rebuilds the
// exact bytes the client sent and the backing backend returned.
func TestRecorderImageRequestResponseRoundTripPreservesPayloads(t *testing.T) {
	// 1x1 red PNG, base64 — realistic generated-image artifact.
	png, err := base64.StdEncoding.DecodeString("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
	if err != nil {
		t.Fatalf("decode fixture png: %v", err)
	}
	artifact := base64.StdEncoding.EncodeToString(png)

	requestBody := `{"model":"image-backend","input":"draw a red fox","tools":[{"type":"image_generation","model":"gpt-image-1"}],"tool_choice":{"type":"image_generation"}}`
	responseBody := `{"created":1701234567,"data":[{"b64_json":"` + artifact + `"},{"b64_json":"` + artifact + `"}]}`

	collect := store.NewMemoryStore(10, 0)
	recorder := NewRecorder(collect)
	recorder.SetCapturePolicy(true, true, 10<<20) // 10 MiB — artifact far under

	id, err := recorder.AddRecord(RoutingRecord{
		RequestID:     "req-image-roundtrip",
		OriginalModel: "default-backend",
		SelectedModel: "image-backend",
		RequestBody:   requestBody,
		ResponseBody:  responseBody,
	})
	if err != nil {
		t.Fatalf("AddRecord: %v", err)
	}

	rec, found := recorder.GetRecord(id)
	if !found {
		t.Fatal("record not found after capture")
	}
	if rec.RequestBody != requestBody {
		t.Fatalf("request body changed across replay round-trip\n got %q\nwant %q", rec.RequestBody, requestBody)
	}
	if rec.ResponseBody != responseBody {
		t.Fatalf("response body changed across replay round-trip\n got %q\nwant %q", rec.ResponseBody, responseBody)
	}
	if rec.RequestBodyTruncated || rec.ResponseBodyTruncated {
		t.Fatalf("image bodies truncated: req=%v resp=%v", rec.RequestBodyTruncated, rec.ResponseBodyTruncated)
	}
	if rec.SelectedModel != "image-backend" {
		t.Fatalf("replay record lost final dispatch model: %q", rec.SelectedModel)
	}
}
