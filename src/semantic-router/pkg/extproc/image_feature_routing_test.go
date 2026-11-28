package extproc

import (
	"bytes"
	"context"
	"encoding/base64"
	"image"
	"image/color"
	"image/png"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing/parity"
)

const imageFeatureRoutingRecipe = `
    projections:
      scores:
        - name: layout-score
          method: weighted_sum
          inputs: [{type: image_features, name: edge_density, weight: 1, value_source: raw}]
      mappings:
        - name: layout-band
          source: layout-score
          outputs: [{name: dense, gte: 0.5}, {name: simple, lt: 0.5}]
`

func TestImageFeatureRoutingSessionSelectsVisionModel(t *testing.T) {
	corpus, _ := loadParityCorpus(t)
	baseline := strings.Replace(string(corpus.Config), "    signals:\n", "    signals:\n      input_modality: [{name: has-image, modality: image}]\n", 1)
	baseline = strings.Replace(baseline, "    decisions:\n", `    decisions:
      - name: image-fallback
        priority: 50
        rules: {type: input_modality, name: has-image}
        modelRefs: [{model: general-small}]
`, 1)
	enabled := strings.Replace(baseline, "    decisions:\n", imageFeatureRoutingRecipe+`    decisions:
      - name: dense-image
        priority: 70
        rules: {type: projection, name: dense, on_unknown: no_match}
        modelRefs: [{model: general-large}]
      - name: simple-image
        priority: 60
        rules: {type: projection, name: simple, on_unknown: no_match}
        modelRefs: [{model: general-small}]
`, 1)
	newRecorder := func(raw string) *parity.Recorder {
		t.Helper()
		cfg, err := config.ParseYAMLBytes([]byte(raw))
		if err != nil {
			t.Fatal(err)
		}
		return parity.NewRecorder(newParityRouter(t, cfg), routing.DefaultOptions)
	}
	on, off := newRecorder(enabled), newRecorder(baseline)
	for _, tc := range []struct {
		name, url, model, decision string
		recorder                   *parity.Recorder
	}{
		{"simple", imageRoutingTestURL(t, false), "general-small", "simple-image", on},
		{"dense", imageRoutingTestURL(t, true), "general-large", "dense-image", on},
		{"unavailable", "https://example.com/image.png", "general-small", "image-fallback", on},
		{"disabled", imageRoutingTestURL(t, true), "general-small", "image-fallback", off},
	} {
		t.Run(tc.name, func(t *testing.T) {
			body := `{"model":"vllm-sr/auto","messages":[{"role":"user","content":[{"type":"text","text":"Describe this image"},{"type":"image_url","image_url":{"url":"` + tc.url + `"}}]},` +
				`{"role":"system","content":"Use a concise answer."}]}`
			record := tc.recorder.Run(context.Background(), parity.Case{
				Name:     tc.name,
				Request:  parity.CaseRequest{Method: "POST", Path: "/v1/chat/completions", Body: body, Headers: [][]string{{"content-type", "application/json"}}},
				Upstream: &parity.Upstream{Status: 200, Headers: [][]string{{"content-type", "application/json"}}, Body: `{"choices":[{"message":{"role":"assistant","content":"Image description"},"finish_reason":"stop"}]}`},
			})
			if record.Error != "" || record.Evidence.Model != tc.model || record.Evidence.Decision != tc.decision {
				t.Fatalf("error=%q model=%q decision=%q; want %q, %q", record.Error, record.Evidence.Model, record.Evidence.Decision, tc.model, tc.decision)
			}
		})
	}
}

func imageRoutingTestURL(t *testing.T, dense bool) string {
	t.Helper()
	im := image.NewGray(image.Rect(0, 0, 256, 256))
	for y := range 256 {
		for x := range 256 {
			v := x / 128
			if dense {
				v = x % 2
			}
			im.SetGray(x, y, color.Gray{Y: uint8(v * 255)})
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, im); err != nil {
		t.Fatal(err)
	}
	return "data:image/png;base64," + base64.StdEncoding.EncodeToString(buf.Bytes())
}
