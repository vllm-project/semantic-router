package contextcompression

import (
	"context"
	"fmt"
	"math"
	"strings"
	"testing"
)

// ratioCounter mirrors the router's calibrated counter at a fixed bytes-per-token ratio.
func ratioCounter(bytesPerToken float64, calibrated bool) CalibratedTokenCounter {
	return CalibratedTokenCounter{Estimate: func(_ string, byteLength int) (int, bool) {
		return max(1, int(math.Ceil(float64(byteLength)/bytesPerToken))), calibrated
	}}
}

type fixedSourceCounter struct {
	bytesPerToken float64
	source        string
}

func (counter fixedSourceCounter) CountText(_ string, text string) (int, string) {
	if counter.bytesPerToken <= 0 {
		return len(text), counter.source
	}
	return int(math.Ceil(float64(len(text)) / counter.bytesPerToken)), counter.source
}

func (counter fixedSourceCounter) CountRequest(model string, request *RequestIR) (int, string) {
	total := 0
	for _, message := range request.Messages {
		for _, block := range message.Blocks {
			count, _ := counter.CountText(model, block.Text)
			total += count
		}
	}
	return total, counter.source
}

func unitTestToolOutput(row string, rows int) string {
	var builder strings.Builder
	for index := range rows {
		if index == rows/2 {
			builder.WriteString("auth validator failed for tenant 42\n")
		}
		fmt.Fprintf(&builder, row, index)
	}
	return builder.String()
}

// The engine estimates log rows at 4 bytes/token and word rows denser than that.
var unitTestToolOutputs = map[string]string{
	"log rows":   unitTestToolOutput("irrelevant billing row %05d status=ok\n", 1200),
	"word dense": unitTestToolOutput("a b c d e %05d\n", 3000),
}

func unitTestToolRequest(content string) (map[string]interface{}, *RequestIR) {
	body := map[string]interface{}{
		"model": "model",
		"messages": []interface{}{
			map[string]interface{}{"role": "user", "content": "why did auth validation fail"},
			map[string]interface{}{"role": "tool", "tool_call_id": "call_1", "content": content},
		},
	}
	return body, ParseRequestIR(body, Provenance{})
}

// The shipped tool-output fragment: extractive, min_tokens 2000, target_tokens 1000.
func shippedToolOutputPolicy() Policy {
	return Policy{
		Mode:    ModeAuto,
		Budget:  Budget{TargetTokens: 1000},
		Targets: Targets{ToolOutputs: TargetPolicy{Mode: TargetExtractive, MinTokens: 2000, TargetTokens: 1000}},
		Scoring: Scoring{Method: "bm25"},
	}
}

func TestToolOutputCompressionIsIndependentOfCounterUnit(t *testing.T) {
	counters := []struct {
		name    string
		counter TokenCounter
	}{
		{"heuristic", HeuristicTokenCounter{}},
		{"cold model_heuristic", ratioCounter(4, false)},
		{"provider_calibrated claude density", ratioCounter(2.48, true)},
		{"provider_calibrated qwen density", ratioCounter(3.80, true)},
		{"provider_calibrated sparse density", ratioCounter(5.5, true)},
		{"exact tokenizer", fixedSourceCounter{bytesPerToken: 3.1, source: "tokenizer"}},
		{"utf8_byte_upper_bound", fixedSourceCounter{source: "utf8_byte_upper_bound"}},
	}
	for fixture, content := range unitTestToolOutputs {
		for _, tc := range counters {
			t.Run(fixture+"/"+tc.name, func(t *testing.T) {
				body, ir := unitTestToolRequest(content)
				before, _ := tc.counter.CountText("model", content)
				if before < 2000 {
					t.Fatalf("fixture counts %d tokens, below min_tokens", before)
				}
				result := NewService().Apply(context.Background(), Request{
					Model: "model", Request: ir, Policy: shippedToolOutputPolicy(), TokenCounter: tc.counter,
				})
				if !result.Applied || result.BlocksCompressed != 1 || result.Plan.SkipReason != SkipNone {
					t.Fatalf("Apply() applied=%v blocks=%d skip=%q, want one compressed block",
						result.Applied, result.BlocksCompressed, result.Plan.SkipReason)
				}
				compressed := body["messages"].([]interface{})[1].(map[string]interface{})["content"].(string)
				after, _ := tc.counter.CountText("model", compressed)
				// The target is in the request's unit; ratio scaling allows 10% for mixed density.
				if after > 1100 || after < 900 {
					t.Fatalf("compressed block counts %d request tokens (from %d), want 900-1100", after, before)
				}
				if !strings.Contains(compressed, "auth validator failed") {
					t.Fatal("relevant tool evidence was not retained")
				}
			})
		}
	}
}

func TestCompressCandidateTextKeepsEngineUnitSourcesUnchanged(t *testing.T) {
	content := unitTestToolOutputs["log rows"]
	candidate := func(counter TokenCounter) plannedCandidate {
		tokens, _ := counter.CountText("model", content)
		return plannedCandidate{
			block: &TextBlockIR{Text: content, Source: TargetToolOutput},
			plan:  TargetPlan{Kind: TargetToolOutput, OriginalTokens: tokens, TargetTokens: 1000, Query: "auth validator"},
		}
	}
	estimated := EstimateTokens(content)
	heuristic := candidate(HeuristicTokenCounter{})
	cold := candidate(ratioCounter(4, false))
	utf8Bytes := candidate(fixedSourceCounter{source: "utf8_byte_upper_bound"})
	cases := []struct {
		name      string
		counter   TokenCounter
		candidate plannedCandidate
		want      Result
	}{
		{
			"heuristic",
			HeuristicTokenCounter{},
			heuristic,
			CompressToolOutput(content, "auth validator", heuristic.plan.OriginalTokens, 1000),
		},
		{
			"cold model_heuristic",
			ratioCounter(4, false),
			cold,
			CompressToolOutput(content, "auth validator", cold.plan.OriginalTokens, 1000),
		},
		{
			"utf8_byte_upper_bound",
			fixedSourceCounter{source: "utf8_byte_upper_bound"},
			utf8Bytes,
			CompressToolOutput(content, "auth validator", estimated, int(1000*float64(estimated)/float64(len(content)))),
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := compressCandidateText("model", tc.counter, tc.candidate)
			if got != tc.want {
				t.Fatalf("compressCandidateText() = %+v, want pre-existing result %+v", got, tc.want)
			}
		})
	}
}

func TestHistoryProseCompressionHonorsDenseCalibratedCounter(t *testing.T) {
	counter := ratioCounter(2.48, true)
	original := "HEADER\n" + strings.Repeat("Archived unrelated text. ", 2000) + "\nEND"
	ir := ParseRequestIR(map[string]interface{}{"messages": []interface{}{
		map[string]interface{}{"role": "user", "content": original},
		map[string]interface{}{"role": "assistant", "content": "First answer."},
		map[string]interface{}{"role": "user", "content": "Use calculator for 997*991."},
	}}, Provenance{})
	result := NewService().Apply(context.Background(), Request{
		Model: "model", Request: ir, TokenCounter: counter,
		Policy: Policy{Mode: ModeAuto, Budget: Budget{TargetTokens: 1000}, Targets: Targets{
			History:     TargetPolicy{Mode: TargetExtractive, MinTokens: 2000, TargetTokens: 500},
			CurrentUser: TargetPolicy{Mode: TargetPreserve}, ToolOutputs: TargetPolicy{Mode: TargetPreserve},
		}},
	})
	after, _ := counter.CountText("model", ir.Messages[0].Blocks[0].Text)
	if !result.Applied || result.Plan.SkipReason != SkipNone || len(result.Plan.Targets) != 1 {
		t.Fatalf("history prose applied=%v skip=%q, want one compressed block", result.Applied, result.Plan.SkipReason)
	}
	if target := result.Plan.Targets[0].TargetTokens; after > target {
		t.Fatalf("history prose counts %d request tokens, want at most target %d", after, target)
	}
}
