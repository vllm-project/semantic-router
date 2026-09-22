package recipe

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"
)

func generatedPatternManifest(fields string) []byte {
	return []byte(`schema_version: v1
name: repeated-text-test
routing_assets:
  yaml: config.yaml
  dsl: recipe.dsl
coverage: {}
decisions:
- id: record
  expected_decision: record
  variants:
  - id: repeated
    messages:
    - role: user
      content:
      - type: text
        text: "问 "
    generated_text:
      message_index: 0
      content_index: 1
      target_text_bytes: 26
` + fields)
}

func TestGeneratedTextPatternPreservesMeaningAndExactBytes(t *testing.T) {
	manifest, err := decodeProbes(generatedPatternManifest("      text: \"route record\\n\"\n"))
	if err != nil {
		t.Fatal(err)
	}
	probes, _ := flattenProbes(manifest)
	probe := probes[0]
	if probe.GeneratedText.Text != "route record\n" || probe.GeneratedText.Character != "" {
		t.Fatalf("generated metadata = %#v", probe.GeneratedText)
	}
	metadata, err := json.Marshal(probe.GeneratedText)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(metadata), `"character"`) || !strings.Contains(string(metadata), `"text":"route record\n"`) {
		t.Fatalf("generated metadata JSON = %s", metadata)
	}
	request, err := materializeEvalRequest(probe)
	if err != nil {
		t.Fatal(err)
	}
	content := request.Messages[0]["content"].([]any)
	if got := content[1].(map[string]any)["text"]; got != "route record\nroute rec" {
		t.Fatalf("repeated content = %q, want full record and exact prefix", got)
	}
	if got := messageContentTextBytes(content); got != 26 {
		t.Fatalf("materialized text bytes = %d, want 26 including explicit UTF-8 text", got)
	}
	if len(probe.Messages[0]["content"].([]any)) != 1 {
		t.Fatal("materialization mutated the compact request")
	}
}

func TestGeneratedTextPatternCountsEscapedRemainderAtByteLimit(t *testing.T) {
	pattern := "<&\t\"\\\n\r"
	for generatedBytes := 1; generatedBytes <= 2*len(pattern)+1; generatedBytes++ {
		t.Run(fmt.Sprint(generatedBytes), func(t *testing.T) {
			probe := ProbeDetail{
				Messages: []map[string]any{{"role": "user", "content": []any{}}},
				GeneratedText: &GeneratedText{
					TargetTextBytes: generatedBytes,
					Text:            pattern,
				},
			}
			messages, err := materializeMessages(probe)
			if err != nil {
				t.Fatal(err)
			}
			encoded, err := json.Marshal(messages)
			if err != nil {
				t.Fatal(err)
			}
			if _, err = materializeMessagesWithLimit(probe, len(encoded)); err != nil {
				t.Fatalf("exact JSON byte limit: %v", err)
			}
			if _, err = materializeMessagesWithLimit(probe, len(encoded)-1); !errors.Is(err, ErrBadRequest) {
				t.Fatalf("one byte below actual encoded size: %v", err)
			}
		})
	}
}

func TestGeneratedTextPatternValidationAndLegacyMetadata(t *testing.T) {
	for _, fields := range []string{
		"      text: \"\"\n",
		"      text: 123\n",
		"      text: null\n",
		"      text: \"é\"\n",
		"      text: \"record\\u000b\"\n",
		"      text: \"record\\u007f\"\n",
		"      text: \"record\\u0000\"\n",
		"      text: record\n      character: x\n",
		"      text: record\n      character: \"\"\n",
		"      text: record\n      unknown: x\n",
		"      character: \"\"\n",
	} {
		t.Run(fmt.Sprintf("%q", fields), func(t *testing.T) {
			if _, err := decodeProbes(generatedPatternManifest(fields)); err == nil {
				t.Fatal("invalid generated text was accepted")
			}
		})
	}
	manifest, err := decodeProbes(generatedPatternManifest(""))
	if err != nil {
		t.Fatal(err)
	}
	probes, _ := flattenProbes(manifest)
	metadata, err := json.Marshal(probes[0].GeneratedText)
	if err != nil {
		t.Fatal(err)
	}
	const legacy = `{"message_index":0,"content_index":1,"target_text_bytes":26,"character":"x"}`
	if string(metadata) != legacy {
		t.Fatalf("legacy metadata changed: %s", metadata)
	}
}
