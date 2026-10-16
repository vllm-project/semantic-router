package llmprotocol

import (
	"math"
	"testing"
)

func TestInputEstimateCountsHistoryMediaToolsWithoutOutput(t *testing.T) {
	request := Request{
		Instructions: []InstructionBlock{{Role: RoleSystem, Content: []Content{{Kind: ContentText, Text: "1234"}}}},
		Messages:     []Message{{Role: RoleTool, Content: []Content{{Kind: ContentToolResult, ToolResult: &ToolResult{Content: []Content{{Kind: ContentText, Text: "5678"}, {Kind: ContentImage, URL: "https://example.invalid/image"}}}}}}},
		Tools:        []Tool{{Name: "read", InputSchema: []byte(`{}`)}},
		Sampling:     Sampling{MaxOutputTokens: Int64(64)},
	}
	estimate := EstimateInput(&request)
	// 12 text bytes/4 + 2 schema bytes + two message frames + tool frame + image.
	want := 3 + 2 + 8 + 8 + InputImageTokens
	if estimate.Tokens != want || !estimate.HasNonText {
		t.Fatalf("estimate=%+v, want tokens=%d", estimate, want)
	}
	request.Sampling.MaxOutputTokens = Int64(4096)
	if EstimateInput(&request) != estimate {
		t.Fatal("generation budget was counted as input")
	}
	if SaturatingTokenSum(math.MaxInt-1, 10) != math.MaxInt {
		t.Fatal("overflow reduced demand")
	}
}

func TestOutputPolicyDefaultAndCap(t *testing.T) {
	fallback, capValue := 64, 100
	for _, test := range []struct {
		name     string
		supplied *int64
		want     int64
	}{
		{"omitted", nil, 64}, {"explicit smaller", Int64(8), 8}, {"explicit larger", Int64(90), 90}, {"existing cap", Int64(200), 100},
	} {
		t.Run(test.name, func(t *testing.T) {
			request := Request{Sampling: Sampling{MaxOutputTokens: test.supplied}}
			DefaultOutputTokens(&request, &fallback)
			CapOutputTokens(&request, &capValue)
			if request.Sampling.MaxOutputTokens == nil || *request.Sampling.MaxOutputTokens != test.want {
				t.Fatalf("output=%v", request.Sampling.MaxOutputTokens)
			}
			if DefaultOutputTokens(&request, &fallback) || CapOutputTokens(&request, &capValue) {
				t.Fatal("second policy application changed bounded call")
			}
		})
	}
	request := Request{}
	CapOutputTokens(&request, &capValue)
	if request.Sampling.MaxOutputTokens != nil {
		t.Fatal("cap invented a default")
	}
}

func TestInputEstimateIncludesConstrainedOutputSchema(t *testing.T) {
	request := Request{Messages: []Message{{Role: RoleUser, Content: []Content{{Kind: ContentText, Text: "1234"}}}}}
	before := EstimateInput(&request)
	request.OutputFormat = OutputFormat{Kind: OutputJSONSchema, Name: "answer", Description: "a typed result", Schema: []byte(`{"type":"object","properties":{"value":{"type":"string"}}}`)}
	after := EstimateInput(&request)
	added := len(request.OutputFormat.Schema) + len(request.OutputFormat.Name) + len(request.OutputFormat.Description)
	if after.Tokens-before.Tokens != added || after.StructuredBytes-before.StructuredBytes != added {
		t.Fatalf("schema demand before=%+v after=%+v", before, after)
	}
}
