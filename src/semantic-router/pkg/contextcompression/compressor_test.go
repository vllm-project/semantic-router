package contextcompression

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestCompressToolOutputKeepsRelevantAndBoundaryChunks(t *testing.T) {
	content := strings.Join([]string{
		"header metadata and source identifier",
		strings.Repeat("unrelated inventory values ", 120),
		"authentication failure in token validator",
		strings.Repeat("unrelated billing records ", 120),
		"footer checksum and provenance",
	}, "\n")

	result := CompressToolOutput(content, "fix authentication token validator", 100, 55)

	if !result.Applied {
		t.Fatal("expected compression to apply")
	}
	if result.CompressedTokens >= result.OriginalTokens {
		t.Fatalf("compression did not reduce tokens: %#v", result)
	}
	for _, expected := range []string{
		"header metadata",
		"authentication failure",
		"footer checksum",
	} {
		if !strings.Contains(result.Content, expected) {
			t.Fatalf("compressed content omitted %q: %s", expected, result.Content)
		}
	}
}

func TestCompressToolOutputPassesThroughBelowThreshold(t *testing.T) {
	content := "small tool output"
	result := CompressToolOutput(content, "small", 100, 50)
	if result.Applied {
		t.Fatal("small content must pass through")
	}
	if result.Content != content {
		t.Fatalf("content changed: %q", result.Content)
	}
}

func TestCompressToolOutputCompressesHugeSingleLineWithinBudget(t *testing.T) {
	content := strings.Repeat("irrelevant inventory values ", 200) +
		"authentication token validator failed " +
		strings.Repeat("unrelated billing records ", 200)

	result := CompressToolOutput(content, "authentication token validator", 100, 60)

	if !result.Applied {
		t.Fatal("expected a huge single-line tool output to be compressed")
	}
	if result.CompressedTokens > 60 {
		t.Fatalf("compressed output exceeded target: %#v", result)
	}
	if !strings.Contains(result.Content, "authentication token validator") {
		t.Fatalf("query-relevant content was removed: %s", result.Content)
	}
}

func TestCompressToolOutputPreservesValidJSONStructure(t *testing.T) {
	payload := map[string]interface{}{
		"status": "authentication token validator failed",
		"logs":   strings.Repeat("x", 6000),
		"nested": map[string]interface{}{
			"code": 500,
			"ok":   false,
		},
	}
	body, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}

	result := CompressToolOutput(string(body), "authentication validator", 100, 160)

	if !result.Applied || result.Format != "json" {
		t.Fatalf("expected JSON-aware compression: %#v", result)
	}
	if result.CompressedTokens > 160 {
		t.Fatalf("compressed JSON exceeded target: %#v", result)
	}
	var decoded map[string]interface{}
	if err := json.Unmarshal([]byte(result.Content), &decoded); err != nil {
		t.Fatalf("compressed content is invalid JSON: %v\n%s", err, result.Content)
	}
	if decoded["status"] != payload["status"] {
		t.Fatalf("relevant status changed: %#v", decoded["status"])
	}
	nested, ok := decoded["nested"].(map[string]interface{})
	if !ok || nested["code"] != float64(500) || nested["ok"] != false {
		t.Fatalf("JSON structure or primitive types changed: %#v", decoded)
	}
}

func TestCompressToolOutputRanksCJKQueryTerms(t *testing.T) {
	content := strings.Join([]string{
		"开始 元数据 来源",
		strings.Repeat("库存记录无关信息", 200),
		"身份验证令牌校验失败需要修复",
		strings.Repeat("账单记录无关信息", 200),
		"结束 校验和",
	}, "\n")

	result := CompressToolOutput(content, "修复身份验证令牌", 100, 90)

	if !result.Applied {
		t.Fatal("expected CJK content compression")
	}
	if !strings.Contains(result.Content, "身份验证令牌") {
		t.Fatalf("CJK query-relevant content was removed: %s", result.Content)
	}
}

func TestCompressToolOutputDeterministicForWhitespaceFreeBlob(t *testing.T) {
	content := strings.Repeat("A", 12000)
	first := CompressToolOutput(content, "blob", 100, 100)
	second := CompressToolOutput(content, "blob", 100, 100)

	if !first.Applied {
		t.Fatal("conservative byte estimate must compress whitespace-free blobs")
	}
	if first.Content != second.Content || first.CompressedTokens != second.CompressedTokens {
		t.Fatal("compression must be deterministic")
	}
}
