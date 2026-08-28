package recipe

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestServiceMaterializesGeneratedTextAsOneMessagePart(t *testing.T) {
	probe := ProbeDetail{
		Messages: []map[string]any{{
			"role": "user",
			"content": []any{
				map[string]any{"type": "text", "text": "ask "},
				map[string]any{"type": "output_text", "text": "ok"},
				map[string]any{"type": "image_fixture", "fixture": "pixel"},
			},
		}},
		GeneratedText: &GeneratedText{
			MessageIndex:    0,
			ContentIndex:    2,
			TargetTextBytes: 11,
			Character:       "x",
		},
		imageFixtures: map[string]probeImageFixture{
			"pixel": {
				MediaType:  "image/png",
				DataBase64: validOnePixelPNGBase64,
			},
		},
	}

	messages, err := materializeMessages(probe)
	if err != nil {
		t.Fatalf("materializeMessages(): %v", err)
	}
	content, ok := messages[0]["content"].([]any)
	if !ok || len(content) != 4 {
		t.Fatalf("materialized content = %#v, want four content parts", messages[0]["content"])
	}
	generated, _ := content[2].(map[string]any)
	if generated["type"] != "text" || generated["text"] != "xxxxx" {
		t.Fatalf("generated content = %#v", generated)
	}
	if got := messageContentTextBytes(content); got != 11 {
		t.Fatalf("materialized text bytes = %d, want 11", got)
	}
	if original := probe.Messages[0]["content"].([]any); len(original) != 3 {
		t.Fatalf("materialization mutated compact probe: %#v", original)
	}
}

func TestChatRequestReservesFinalPlaygroundEnvelopeAtByteLimit(t *testing.T) {
	probe := ProbeDetail{
		ProbeSummary: ProbeSummary{Model: "vllm-sr/test"},
		Messages: []map[string]any{{
			"role":    "user",
			"content": []any{},
		}},
		GeneratedText: &GeneratedText{
			MessageIndex:    0,
			ContentIndex:    0,
			TargetTextBytes: 1,
			Character:       "x",
		},
	}

	request, err := materializeChatRequest(probe)
	if err != nil {
		t.Fatalf("materializeChatRequest(): %v", err)
	}
	encoded, err := json.Marshal(request)
	if err != nil {
		t.Fatal(err)
	}
	probe.GeneratedText.TargetTextBytes += maxRequestBytes - len(encoded)

	request, err = materializeChatRequest(probe)
	if err != nil {
		t.Fatalf("exact-boundary materializeChatRequest(): %v", err)
	}
	encoded, err = json.Marshal(request)
	if err != nil {
		t.Fatal(err)
	}
	if len(encoded) != maxRequestBytes {
		t.Fatalf("request bytes = %d, want %d", len(encoded), maxRequestBytes)
	}
	if !request.Stream || request.MaxCompletionTokens != defaultCompletionTokens {
		t.Fatalf("request transport policy = %#v", request)
	}

	probe.GeneratedText.TargetTextBytes++
	if _, err := materializeChatRequest(probe); !errors.Is(err, ErrBadRequest) {
		t.Fatalf("over-boundary materializeChatRequest() error = %v, want ErrBadRequest", err)
	}
}

func TestProbeManifestRejectsGeneratedTextBelowExplicitText(t *testing.T) {
	_, err := decodeProbes([]byte(`schema_version: v1
name: generated-text-test
routing_assets:
  yaml: config.yaml
  dsl: recipe.dsl
coverage: {}
decisions:
  - id: long-context
    expected_decision: long-context
    variants:
      - id: invalid
        messages:
          - role: user
            content:
              - type: text
                text: explicit
        generated_text:
          message_index: 0
          content_index: 1
          target_text_bytes: 8
`))
	if err == nil || !strings.Contains(err.Error(), "must exceed the selected message's 8 explicit text bytes") {
		t.Fatalf("decodeProbes() error = %v", err)
	}
}

func TestProbeVariantEditableRequiresOneTerminalTextPart(t *testing.T) {
	zero := 0
	tests := []struct {
		name    string
		variant probeVariant
		want    bool
	}{
		{
			name: "plain text",
			variant: probeVariant{Messages: []map[string]any{{
				"role": "user", "content": "edit me",
			}}},
			want: true,
		},
		{
			name: "one text plus image fixture",
			variant: probeVariant{Messages: []map[string]any{{
				"role": "user",
				"content": []any{
					map[string]any{"type": "text", "text": "edit me"},
					map[string]any{"type": "image_fixture", "fixture": "diagram"},
				},
			}}},
			want: true,
		},
		{
			name: "multiple text parts",
			variant: probeVariant{Messages: []map[string]any{{
				"role": "user",
				"content": []any{
					map[string]any{"type": "text", "text": "first"},
					map[string]any{"type": "text", "text": "second"},
				},
			}}},
		},
		{
			name: "generated terminal text",
			variant: probeVariant{
				Messages: []map[string]any{{
					"role":    "user",
					"content": []any{map[string]any{"type": "text", "text": "prefix"}},
				}},
				GeneratedText: &probeGeneratedText{MessageIndex: &zero},
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := probeVariantIsEditable(test.variant); got != test.want {
				t.Fatalf("probeVariantIsEditable() = %t, want %t", got, test.want)
			}
		})
	}
}

func TestProbeImageFixtureMaterializesVerifiedDataURIAndMetadata(t *testing.T) {
	manifest, err := decodeProbes([]byte(validImageFixtureManifest(validOnePixelPNGBase64, validOnePixelPNGSHA256, "pixel")))
	if err != nil {
		t.Fatalf("decodeProbes(): %v", err)
	}
	probes, _ := flattenProbes(manifest)
	probe := probes[0]
	messages, err := materializeMessages(probe)
	if err != nil {
		t.Fatalf("materializeMessages(): %v", err)
	}
	content := messages[0]["content"].([]any)
	image := content[1].(map[string]any)["image_url"].(map[string]any)
	if image["url"] != "data:image/png;base64,"+validOnePixelPNGBase64 {
		t.Fatalf("materialized image fixture = %#v", content[1])
	}
	chatRequest, err := materializeChatRequest(probe)
	if err != nil {
		t.Fatalf("materializeChatRequest(): %v", err)
	}
	evalRequest, err := materializeEvalRequest(probe)
	if err != nil {
		t.Fatalf("materializeEvalRequest(): %v", err)
	}
	chatMessages, _ := json.Marshal(chatRequest.Messages)
	evalMessages, _ := json.Marshal(evalRequest.Messages)
	if string(chatMessages) != string(evalMessages) {
		t.Fatalf("Run and Validate materialization drifted:\nrun: %s\nvalidate: %s", chatMessages, evalMessages)
	}
	metadata := probe.ImageFixtures["pixel"]
	if metadata.Description != "One-pixel fixture for loader tests." || metadata.MediaType != "image/png" ||
		metadata.Bytes != 68 || metadata.SHA256 != validOnePixelPNGSHA256 {
		t.Fatalf("image fixture metadata = %#v", metadata)
	}
	encoded, err := json.Marshal(probe)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(encoded), "data_base64") || strings.Contains(string(encoded), validOnePixelPNGBase64) {
		t.Fatalf("probe detail exposed fixture payload: %s", encoded)
	}
	cloned := cloneProbeDetail(probe)
	cloned.ImageFixtures["pixel"] = ImageFixtureMetadata{Description: "mutated"}
	if probe.ImageFixtures["pixel"].Description != "One-pixel fixture for loader tests." {
		t.Fatal("cloneProbeDetail shared image fixture metadata map")
	}
}

func validImageFixtureManifest(dataBase64, fixtureSHA256, reference string) string {
	return fmt.Sprintf(`schema_version: v1
name: image-fixture-test
routing_assets:
  yaml: config.yaml
  dsl: recipe.dsl
fixtures:
  images:
    pixel:
      description: One-pixel fixture for loader tests.
      media_type: image/png
      data_base64: %s
      sha256: %s
coverage: {}
decisions:
  - id: vision
    expected_decision: vision
    variants:
      - id: fixture
        messages:
          - role: user
            content:
              - type: text
                text: inspect
              - type: image_fixture
                fixture: %s
`, dataBase64, fixtureSHA256, reference)
}

func TestProbeImageFixtureRejectsUnknownHashAndNoncanonicalBase64(t *testing.T) {
	tests := []struct {
		name       string
		dataBase64 string
		sha256     string
		reference  string
		want       string
	}{
		{name: "unknown reference", dataBase64: validOnePixelPNGBase64, sha256: validOnePixelPNGSHA256, reference: "missing", want: "unknown image fixture"},
		{name: "numeric reference", dataBase64: validOnePixelPNGBase64, sha256: validOnePixelPNGSHA256, reference: "1", want: "must be a non-empty string"},
		{name: "hash drift", dataBase64: validOnePixelPNGBase64, sha256: "sha256:" + strings.Repeat("0", 64), reference: "pixel", want: "sha256 mismatch"},
		{name: "noncanonical pad bits", dataBase64: "AB==", sha256: validOnePixelPNGSHA256, reference: "pixel", want: "data_base64 is invalid"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := decodeProbes([]byte(validImageFixtureManifest(test.dataBase64, test.sha256, test.reference)))
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("decodeProbes() error = %v, want %q", err, test.want)
			}
		})
	}
}

func TestProbeImageFixtureTypeRequiresCanonicalLowercase(t *testing.T) {
	valid := validImageFixtureManifest(validOnePixelPNGBase64, validOnePixelPNGSHA256, "pixel")
	for _, itemType := range []string{"IMAGE_FIXTURE", `" image_fixture "`} {
		t.Run(itemType, func(t *testing.T) {
			manifest := strings.Replace(valid, "type: image_fixture", "type: "+itemType, 1)
			_, err := decodeProbes([]byte(manifest))
			if err == nil || !strings.Contains(err.Error(), "type must be exactly") {
				t.Fatalf("decodeProbes() error = %v", err)
			}
		})
	}

	probe := ProbeDetail{
		Messages: []map[string]any{{
			"role": "user",
			"content": []any{
				map[string]any{"type": "IMAGE_FIXTURE", "fixture": "pixel"},
			},
		}},
		imageFixtures: map[string]probeImageFixture{
			"pixel": {MediaType: "image/png", DataBase64: validOnePixelPNGBase64},
		},
	}
	_, err := materializeMessages(probe)
	if err == nil || !strings.Contains(err.Error(), "type must be exactly") {
		t.Fatalf("materializeMessages() error = %v", err)
	}
}

func TestProbeManifestRejectsRawImageSources(t *testing.T) {
	tests := []struct {
		name      string
		item      string
		forbidden string
	}{
		{
			name:      "metadata endpoint",
			item:      `{type: image_url, image_url: {url: "http://169.254.169.254/latest/meta-data"}}`,
			forbidden: "169.254.169.254",
		},
		{
			name:      "localhost",
			item:      `{type: input_image, image_url: "http://127.0.0.1/private"}`,
			forbidden: "127.0.0.1",
		},
		{
			name:      "file URL",
			item:      `{type: image_url, image_url: {url: "file:///etc/passwd"}}`,
			forbidden: "/etc/passwd",
		},
		{
			name:      "inline data URL",
			item:      `{type: image_url, image_url: {url: "data:image/png;base64,AA=="}}`,
			forbidden: "AA==",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			manifest := fmt.Sprintf(`schema_version: v1
name: raw-image-test
routing_assets:
  yaml: config.yaml
  dsl: recipe.dsl
coverage: {}
decisions:
  - id: vision
    expected_decision: vision
    variants:
      - id: raw
        messages:
          - role: user
            content:
              - type: text
                text: inspect
              - %s
`, test.item)
			_, err := decodeProbes([]byte(manifest))
			if err == nil || !strings.Contains(err.Error(), "must use a declared image_fixture") {
				t.Fatalf("decodeProbes() error = %v", err)
			}
			if strings.Contains(err.Error(), test.forbidden) {
				t.Fatalf("decodeProbes() leaked rejected image source: %v", err)
			}
		})
	}
}

func TestMaterializeMessagesDefensivelyRejectsRawImageSource(t *testing.T) {
	probe := ProbeDetail{Messages: []map[string]any{{
		"role": "user",
		"content": []any{
			map[string]any{"type": "text", "text": "inspect"},
			map[string]any{
				"type":      "image_url",
				"image_url": map[string]any{"url": "https://private.invalid/image.png"},
			},
		},
	}}}
	_, err := materializeMessages(probe)
	if err == nil || !strings.Contains(err.Error(), "must use a declared image_fixture") {
		t.Fatalf("materializeMessages() error = %v", err)
	}
	if strings.Contains(err.Error(), "private.invalid") {
		t.Fatalf("materializeMessages() leaked rejected image source: %v", err)
	}
}

func TestEvalRequestRejectsFixtureExpansionBeforeMaterialization(t *testing.T) {
	probe := ProbeDetail{
		ProbeSummary: ProbeSummary{Model: "test/model"},
		Messages: []map[string]any{{
			"role": "user",
			"content": []any{
				map[string]any{"type": "text", "text": "inspect"},
				map[string]any{"type": "image_fixture", "fixture": "pixel"},
			},
		}},
		imageFixtures: map[string]probeImageFixture{
			"pixel": {
				MediaType:  "image/png",
				DataBase64: strings.Repeat("A", 4096),
			},
		},
	}

	_, err := materializeEvalRequestWithLimit(probe, 256)
	if !errors.Is(err, ErrBadRequest) || !strings.Contains(err.Error(), "materialized messages exceed") {
		t.Fatalf("materializeEvalRequestWithLimit() error = %v", err)
	}
	content := probe.Messages[0]["content"].([]any)
	if fixture := content[1].(map[string]any); fixture["type"] != "image_fixture" {
		t.Fatalf("failed preflight mutated compact fixture: %#v", fixture)
	}
}

func TestProbeManifestRejectsEmptyFixtureCatalog(t *testing.T) {
	issues := []string{}
	validateProbeFixtures(&probeFixtures{}, &issues)
	if len(issues) != 1 || !strings.Contains(issues[0], "must contain at least one") {
		t.Fatalf("fixture validation issues = %#v", issues)
	}
}

func TestProbeManifestRejectsYAMLIndirection(t *testing.T) {
	tests := []struct {
		name     string
		document string
		want     string
	}{
		{name: "anchor", document: "value: &hidden 1\n", want: "anchors"},
		{name: "alias", document: "value: &hidden 1\nother: *hidden\n", want: "anchors"},
		{name: "tag", document: "value: !!str hidden\n", want: "tags"},
		{name: "merge", document: "value:\n  <<: {}\n", want: "merge keys"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := decodeProbes([]byte(test.document))
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("decodeProbes() error = %v, want %q", err, test.want)
			}
		})
	}
}

type momFixtureReceipt struct {
	probeCount      int
	messageProbes   int
	generatedProbes int
	imageParts      int
	imageURLs       map[string]struct{}
	textBytes       int
	textDigest      string
}

func TestMoMGeneratedTextPreservesLegacyTextReceipt(t *testing.T) {
	path := filepath.Join("..", "..", "..", "config", "recipes", "built-in", "latest", "mom-v1", "probes.yaml")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile(%s): %v", path, err)
	}
	manifest, err := decodeProbes(data)
	if err != nil {
		t.Fatalf("decodeProbes(%s): %v", path, err)
	}
	receipt := materializeMoMFixtureReceipt(t, manifest)
	if receipt.probeCount != 226 || receipt.messageProbes != 85 || receipt.generatedProbes != 50 ||
		receipt.imageParts != 57 || receipt.textBytes != 26_229_731 {
		t.Fatalf("receipt counts = %#v", receipt)
	}
	if receipt.textDigest != "17e56c0457f3ffd9419a25ad8b8e0de60c21fbcf62a88d31fced47c2d1a189b7" {
		t.Fatalf("materialized text digest = %s", receipt.textDigest)
	}
	assertMoMImageFixtureReceipt(t, manifest, receipt.imageURLs)
}

func materializeMoMFixtureReceipt(t *testing.T, manifest probeManifest) momFixtureReceipt {
	t.Helper()
	probes, _ := flattenProbes(manifest)
	receipt := momFixtureReceipt{probeCount: len(probes), imageURLs: map[string]struct{}{}}
	digest := sha256.New()
	for _, probe := range probes {
		if len(probe.Messages) == 0 {
			continue
		}
		receipt.messageProbes++
		if probe.GeneratedText != nil {
			receipt.generatedProbes++
		}
		messages, err := materializeMessages(probe)
		if err != nil {
			t.Fatalf("materializeMessages(%s): %v", probe.ID, err)
		}
		payload := probeTextPayload(messages)
		writeProbeReceipt(digest, probe.ID, payload)
		receipt.textBytes += len(payload)
		for _, message := range messages {
			collectImageURLs(message["content"], receipt.imageURLs, &receipt.imageParts)
		}
	}
	receipt.textDigest = hex.EncodeToString(digest.Sum(nil))
	return receipt
}

func probeTextPayload(messages []map[string]any) []byte {
	var payload []byte
	for _, message := range messages {
		switch content := message["content"].(type) {
		case string:
			payload = append(payload, content...)
		case []any:
			for _, raw := range content {
				item, ok := raw.(map[string]any)
				if !ok {
					continue
				}
				kind, _ := item["type"].(string)
				kind = strings.ToLower(strings.TrimSpace(kind))
				if kind == "" || kind == "text" || kind == "input_text" || kind == "output_text" {
					text, _ := item["text"].(string)
					payload = append(payload, text...)
				}
			}
		}
	}
	return payload
}

func writeProbeReceipt(digest interface{ Write([]byte) (int, error) }, probeID string, payload []byte) {
	_, _ = digest.Write([]byte(probeID))
	_, _ = digest.Write([]byte{0})
	var length [8]byte
	binary.BigEndian.PutUint64(length[:], uint64(len(payload)))
	_, _ = digest.Write(length[:])
	_, _ = digest.Write(payload)
}

func collectImageURLs(content any, urls map[string]struct{}, count *int) {
	items, ok := content.([]any)
	if !ok {
		return
	}
	for _, raw := range items {
		item, ok := raw.(map[string]any)
		if !ok || item["type"] != "image_url" {
			continue
		}
		*count++
		imageURL, _ := item["image_url"].(map[string]any)
		url, _ := imageURL["url"].(string)
		urls[url] = struct{}{}
	}
}

func assertMoMImageFixtureReceipt(
	t *testing.T,
	manifest probeManifest,
	imageURLs map[string]struct{},
) {
	t.Helper()
	if len(imageURLs) != 1 {
		t.Fatalf("materialized image URLs = %d, want 1", len(imageURLs))
	}
	for url := range imageURLs {
		const expectedURLSHA = "c5506a9f23e55bafd166bd07c682566c21363a25c1433cd6e92bf22d3e28a1b3"
		if actual := fmt.Sprintf("%x", sha256.Sum256([]byte(url))); actual != expectedURLSHA {
			t.Fatalf("materialized image URL digest = %s, want %s", actual, expectedURLSHA)
		}
	}
	fixture := manifest.Fixtures.Images["mom_v1_architecture"]
	if fixture.Description != "Compact Client-to-Router-to-Backend diagram used only to exercise image payload shape and vision-routing branches." {
		t.Fatalf("architecture fixture description = %q", fixture.Description)
	}
	if fixture.SHA256 != "sha256:76f8c378648b27757ac72a7ea00f32cc76d4a4240c6e84002a182a0c2de5fde3" {
		t.Fatalf("architecture fixture digest = %q", fixture.SHA256)
	}
}
