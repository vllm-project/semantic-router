package looper

import (
	"fmt"
	"strings"
	"testing"
)

// helper: given input text and the boolean mask, return only the semantic chars.
func semanticChars(text string, mask []bool) string {
	var b strings.Builder
	for i, c := range text {
		if i < len(mask) && mask[i] {
			b.WriteRune(c)
		}
	}
	return b.String()
}

// ---------------------------------------------------------------------------
// classifyToolCallChars – core state-machine tests
// ---------------------------------------------------------------------------

func TestClassifyToolCallChars_SimpleClick(t *testing.T) {
	input := `{"name":"computer_action","arguments":{"action":"click","coordinate":[499,161]}}`
	mask := classifyToolCallChars(input)
	if mask == nil {
		t.Fatal("expected non-nil mask")
	}
	got := semanticChars(input, mask)
	// "click" + "499" + "161" (without quotes or brackets)
	if got != "click499161" {
		t.Errorf("semantic chars = %q, want %q", got, "click499161")
	}
}

func TestClassifyToolCallChars_TypeAction(t *testing.T) {
	input := `{"name":"computer_action","arguments":{"action":"type","text":"hello world"}}`
	mask := classifyToolCallChars(input)
	if mask == nil {
		t.Fatal("expected non-nil mask")
	}
	got := semanticChars(input, mask)
	if got != "typehello world" {
		t.Errorf("semantic chars = %q, want %q", got, "typehello world")
	}
}

func TestClassifyToolCallChars_ScrollAction(t *testing.T) {
	input := `{"name":"computer_action","arguments":{"action":"scroll","coordinate":[512,384],"direction":"down","amount":3}}`
	mask := classifyToolCallChars(input)
	if mask == nil {
		t.Fatal("expected non-nil mask")
	}
	got := semanticChars(input, mask)
	// action=scroll, coordinate 512,384, direction=down, amount=3
	if got != "scroll512384down3" {
		t.Errorf("semantic chars = %q, want %q", got, "scroll512384down3")
	}
}

func TestClassifyToolCallChars_KeyAction(t *testing.T) {
	input := `{"name":"computer_action","arguments":{"action":"key","key":"ctrl+c"}}`
	mask := classifyToolCallChars(input)
	if mask == nil {
		t.Fatal("expected non-nil mask")
	}
	got := semanticChars(input, mask)
	if got != "keyctrl+c" {
		t.Errorf("semantic chars = %q, want %q", got, "keyctrl+c")
	}
}

func TestClassifyToolCallChars_XMLWrapper(t *testing.T) {
	input := `<tool_call>{"name":"computer_action","arguments":{"action":"click","coordinate":[100,200]}}</tool_call>`
	mask := classifyToolCallChars(input)
	if mask == nil {
		t.Fatal("expected non-nil mask for XML-wrapped tool call")
	}
	got := semanticChars(input, mask)
	if got != "click100200" {
		t.Errorf("semantic chars = %q, want %q", got, "click100200")
	}
}

func TestClassifyToolCallChars_NoArguments(t *testing.T) {
	input := `{"name":"get_status","parameters":{"id":"abc"}}`
	mask := classifyToolCallChars(input)
	// No "arguments" key → nothing is semantic → returns nil
	if mask != nil {
		got := semanticChars(input, mask)
		t.Errorf("expected nil mask for no-arguments JSON, got semantic=%q", got)
	}
}

func TestClassifyToolCallChars_EmptyArguments(t *testing.T) {
	input := `{"name":"noop","arguments":{}}`
	mask := classifyToolCallChars(input)
	// arguments is present but empty → no semantic chars → nil
	if mask != nil {
		got := semanticChars(input, mask)
		t.Errorf("expected nil mask for empty arguments, got semantic=%q", got)
	}
}

func TestClassifyToolCallChars_NotJSON(t *testing.T) {
	input := `This is just plain text with no JSON`
	mask := classifyToolCallChars(input)
	if mask != nil {
		t.Error("expected nil mask for non-JSON input")
	}
}

func TestClassifyToolCallChars_EmptyString(t *testing.T) {
	mask := classifyToolCallChars("")
	if mask != nil {
		t.Error("expected nil mask for empty string")
	}
}

func TestClassifyToolCallChars_BooleanAndNullValues(t *testing.T) {
	input := `{"name":"set_flags","arguments":{"verbose":true,"dry_run":false,"extra":null}}`
	mask := classifyToolCallChars(input)
	if mask == nil {
		t.Fatal("expected non-nil mask")
	}
	got := semanticChars(input, mask)
	if got != "truefalsenull" {
		t.Errorf("semantic chars = %q, want %q", got, "truefalsenull")
	}
}

func TestClassifyToolCallChars_NestedObject(t *testing.T) {
	input := `{"name":"complex","arguments":{"outer":{"inner_key":"inner_val"},"simple":"val"}}`
	mask := classifyToolCallChars(input)
	if mask == nil {
		t.Fatal("expected non-nil mask")
	}
	got := semanticChars(input, mask)
	// inner_val and val are argument values; inner_key is a key inside nested obj
	if got != "inner_valval" {
		t.Errorf("semantic chars = %q, want %q", got, "inner_valval")
	}
}

func TestClassifyToolCallChars_MultipleArrayElements(t *testing.T) {
	input := `{"name":"draw","arguments":{"points":[10,20,30,40,50]}}`
	mask := classifyToolCallChars(input)
	if mask == nil {
		t.Fatal("expected non-nil mask")
	}
	got := semanticChars(input, mask)
	if got != "1020304050" {
		t.Errorf("semantic chars = %q, want %q", got, "1020304050")
	}
}

func TestClassifyToolCallChars_StringArray(t *testing.T) {
	input := `{"name":"tag","arguments":{"tags":["alpha","beta","gamma"]}}`
	mask := classifyToolCallChars(input)
	if mask == nil {
		t.Fatal("expected non-nil mask")
	}
	got := semanticChars(input, mask)
	if got != "alphabetagamma" {
		t.Errorf("semantic chars = %q, want %q", got, "alphabetagamma")
	}
}

func TestClassifyToolCallChars_MixedArray(t *testing.T) {
	input := `{"name":"mixed","arguments":{"items":["str",42,true,null]}}`
	mask := classifyToolCallChars(input)
	if mask == nil {
		t.Fatal("expected non-nil mask")
	}
	got := semanticChars(input, mask)
	if got != "str42truenull" {
		t.Errorf("semantic chars = %q, want %q", got, "str42truenull")
	}
}

func TestClassifyToolCallChars_EscapedStringInArgValue(t *testing.T) {
	input := `{"name":"run","arguments":{"cmd":"echo \"hello\""}}`
	mask := classifyToolCallChars(input)
	if mask == nil {
		t.Fatal("expected non-nil mask")
	}
	got := semanticChars(input, mask)
	// The backslashes and inner quotes are part of the escaped content
	if got != `echo \"hello\"` {
		t.Errorf("semantic chars = %q, want %q", got, `echo \"hello\"`)
	}
}

func TestClassifyToolCallChars_NegativeCoordinates(t *testing.T) {
	input := `{"name":"move","arguments":{"x":-10,"y":-20}}`
	mask := classifyToolCallChars(input)
	if mask == nil {
		t.Fatal("expected non-nil mask")
	}
	got := semanticChars(input, mask)
	if got != "-10-20" {
		t.Errorf("semantic chars = %q, want %q", got, "-10-20")
	}
}

func TestClassifyToolCallChars_FloatCoordinates(t *testing.T) {
	input := `{"name":"precise","arguments":{"x":12.5,"y":99.99}}`
	mask := classifyToolCallChars(input)
	if mask == nil {
		t.Fatal("expected non-nil mask")
	}
	got := semanticChars(input, mask)
	if got != "12.599.99" {
		t.Errorf("semantic chars = %q, want %q", got, "12.599.99")
	}
}

func TestClassifyToolCallChars_WhitespaceFormatted(t *testing.T) {
	input := `{
  "name": "computer_action",
  "arguments": {
    "action": "click",
    "coordinate": [640, 480]
  }
}`
	mask := classifyToolCallChars(input)
	if mask == nil {
		t.Fatal("expected non-nil mask")
	}
	got := semanticChars(input, mask)
	if got != "click640480" {
		t.Errorf("semantic chars = %q, want %q", got, "click640480")
	}
}

func TestClassifyToolCallChars_ArgumentsAfterOtherKeys(t *testing.T) {
	// "arguments" is not the second key — there's extra metadata before it
	input := `{"id":"call_001","type":"function","name":"browser_click","arguments":{"selector":"#submit","x":300,"y":450}}`
	mask := classifyToolCallChars(input)
	if mask == nil {
		t.Fatal("expected non-nil mask")
	}
	got := semanticChars(input, mask)
	if got != "#submit300450" {
		t.Errorf("semantic chars = %q, want %q", got, "#submit300450")
	}
}

func TestClassifyToolCallChars_EmptyStringArgValue(t *testing.T) {
	input := `{"name":"type","arguments":{"text":"","target":"input"}}`
	mask := classifyToolCallChars(input)
	if mask == nil {
		t.Fatal("expected non-nil mask")
	}
	got := semanticChars(input, mask)
	// empty string contributes nothing; "input" is the only semantic content
	if got != "input" {
		t.Errorf("semantic chars = %q, want %q", got, "input")
	}
}

func TestClassifyToolCallChars_NestedArrayOfArrays(t *testing.T) {
	input := `{"name":"polygon","arguments":{"coords":[[1,2],[3,4],[5,6]]}}`
	mask := classifyToolCallChars(input)
	if mask == nil {
		t.Fatal("expected non-nil mask")
	}
	got := semanticChars(input, mask)
	if got != "123456" {
		t.Errorf("semantic chars = %q, want %q", got, "123456")
	}
}

func TestClassifyToolCallChars_UnicodeInArgValue(t *testing.T) {
	input := `{"name":"input","arguments":{"text":"日本語テスト"}}`
	mask := classifyToolCallChars(input)
	if mask == nil {
		t.Fatal("expected non-nil mask")
	}
	got := semanticChars(input, mask)
	if got != "日本語テスト" {
		t.Errorf("semantic chars = %q, want %q", got, "日本語テスト")
	}
}

func TestClassifyToolCallChars_LargeCoordinateValues(t *testing.T) {
	input := `{"name":"computer_action","arguments":{"action":"click","coordinate":[1920,1080]}}`
	mask := classifyToolCallChars(input)
	if mask == nil {
		t.Fatal("expected non-nil mask")
	}
	got := semanticChars(input, mask)
	if got != "click19201080" {
		t.Errorf("semantic chars = %q, want %q", got, "click19201080")
	}
}

func TestClassifyToolCallChars_SingleCoordinate(t *testing.T) {
	input := `{"name":"scroll","arguments":{"y":500}}`
	mask := classifyToolCallChars(input)
	if mask == nil {
		t.Fatal("expected non-nil mask")
	}
	got := semanticChars(input, mask)
	if got != "500" {
		t.Errorf("semantic chars = %q, want %q", got, "500")
	}
}

func TestClassifyToolCallChars_NameFieldNotSemantic(t *testing.T) {
	input := `{"name":"computer_action","arguments":{"action":"click","coordinate":[100,200]}}`
	mask := classifyToolCallChars(input)
	if mask == nil {
		t.Fatal("expected non-nil mask")
	}
	// "computer_action" in the name field must NOT be semantic
	nameStart := strings.Index(input, "computer_action")
	for i := nameStart; i < nameStart+len("computer_action"); i++ {
		if mask[i] {
			t.Errorf("char %d (%c) inside 'name' value should not be semantic", i, input[i])
			break
		}
	}
}

func TestClassifyToolCallChars_ArgumentKeysNotSemantic(t *testing.T) {
	input := `{"name":"act","arguments":{"action":"click","coordinate":[1,2]}}`
	mask := classifyToolCallChars(input)
	if mask == nil {
		t.Fatal("expected non-nil mask")
	}
	// Keys "action" and "coordinate" inside arguments must NOT be semantic.
	// Find the key "action" inside arguments (after "arguments":{")
	argStart := strings.Index(input, `"arguments"`)
	keyPos := strings.Index(input[argStart+12:], `"action"`) + argStart + 12
	// The key characters (a,c,t,i,o,n) should not be marked
	for i := keyPos + 1; i < keyPos+7; i++ { // skip opening quote
		if mask[i] {
			t.Errorf("char %d (%c) is an argument key char and should not be semantic", i, input[i])
			break
		}
	}
}

func TestClassifyToolCallChars_MaskLengthMatchesInput(t *testing.T) {
	input := `{"name":"foo","arguments":{"k":"v"}}`
	mask := classifyToolCallChars(input)
	if mask == nil {
		t.Fatal("expected non-nil mask")
	}
	if len(mask) != len(input) {
		t.Errorf("mask length %d != input length %d", len(mask), len(input))
	}
}

func TestClassifyToolCallChars_OnlyBracesNoArguments(t *testing.T) {
	input := `{"hello":"world"}`
	mask := classifyToolCallChars(input)
	if mask != nil {
		t.Error("expected nil mask for JSON without 'arguments' key")
	}
}

func TestClassifyToolCallChars_DeeplyNestedArguments(t *testing.T) {
	input := `{"name":"deep","arguments":{"a":{"b":{"c":"deepval"}}}}`
	mask := classifyToolCallChars(input)
	if mask == nil {
		t.Fatal("expected non-nil mask")
	}
	got := semanticChars(input, mask)
	if got != "deepval" {
		t.Errorf("semantic chars = %q, want %q", got, "deepval")
	}
}

func TestClassifyToolCallChars_ArrayOfObjects(t *testing.T) {
	input := `{"name":"multi","arguments":{"steps":[{"x":1,"y":2},{"x":3,"y":4}]}}`
	mask := classifyToolCallChars(input)
	if mask == nil {
		t.Fatal("expected non-nil mask")
	}
	got := semanticChars(input, mask)
	if got != "1234" {
		t.Errorf("semantic chars = %q, want %q", got, "1234")
	}
}

func TestClassifyToolCallChars_EmptyArray(t *testing.T) {
	input := `{"name":"noop","arguments":{"items":[]}}`
	mask := classifyToolCallChars(input)
	// Empty array means no semantic content → nil
	if mask != nil {
		got := semanticChars(input, mask)
		t.Errorf("expected nil mask for empty array arg, got semantic=%q", got)
	}
}

func TestClassifyToolCallChars_MultipleToolCallsFirstOnly(t *testing.T) {
	// The function processes from the first '{' to the end; a second
	// tool call appended should still parse correctly for the first one.
	input := `{"name":"a","arguments":{"v":1}}{"name":"b","arguments":{"v":2}}`
	mask := classifyToolCallChars(input)
	if mask == nil {
		t.Fatal("expected non-nil mask")
	}
	got := semanticChars(input, mask)
	// Both tool calls should have their argument values marked
	if !strings.Contains(got, "1") {
		t.Errorf("expected '1' in semantic chars, got %q", got)
	}
}

// ---------------------------------------------------------------------------
// filterToolCallArgTokens – integration with token boundaries
// ---------------------------------------------------------------------------

func TestFilterToolCallArgTokens_BasicClick(t *testing.T) {
	resp := &ModelResponse{
		Tokens: []string{
			`{"name":"`, `computer_action`, `","arguments":{"action":"`, `click`,
			`","coordinate":[`, `499`, `,`, `161`, `]}}`,
		},
		Logprobs: []float64{
			-0.01, -0.02, -0.01, -0.30,
			-0.01, -0.80, -0.01, -0.90, -0.01,
		},
		TopLogprobMargins: []float64{
			0.99, 0.98, 0.99, 0.70,
			0.99, 0.20, 0.99, 0.10, 0.99,
		},
	}

	filterToolCallArgTokens(resp)

	if resp.FilteredAverageLogprob == 0 {
		t.Fatal("FilteredAverageLogprob should be non-zero after filtering")
	}

	// Semantic tokens: "click" (idx 3), "499" (idx 5), "161" (idx 7)
	// Their logprobs: -0.30, -0.80, -0.90 → avg = -0.6667
	expectedAvg := (-0.30 + -0.80 + -0.90) / 3.0
	if abs(resp.FilteredAverageLogprob-expectedAvg) > 0.001 {
		t.Errorf("FilteredAverageLogprob = %.4f, want %.4f", resp.FilteredAverageLogprob, expectedAvg)
	}

	expectedMargin := (0.70 + 0.20 + 0.10) / 3.0
	if abs(resp.FilteredAverageMargin-expectedMargin) > 0.001 {
		t.Errorf("FilteredAverageMargin = %.4f, want %.4f", resp.FilteredAverageMargin, expectedMargin)
	}
}

func TestFilterToolCallArgTokens_NoTokens(t *testing.T) {
	resp := &ModelResponse{}
	filterToolCallArgTokens(resp)
	if resp.FilteredAverageLogprob != 0 {
		t.Errorf("expected zero FilteredAverageLogprob for empty tokens, got %f", resp.FilteredAverageLogprob)
	}
}

func TestFilterToolCallArgTokens_NoArgumentsKey(t *testing.T) {
	resp := &ModelResponse{
		Tokens:   []string{`{"result":`, `"ok"`, `}`},
		Logprobs: []float64{-0.01, -0.05, -0.01},
	}
	filterToolCallArgTokens(resp)
	if resp.FilteredAverageLogprob != 0 {
		t.Errorf("expected zero FilteredAverageLogprob when no arguments key, got %f", resp.FilteredAverageLogprob)
	}
}

func TestFilterToolCallArgTokens_AllSemanticFiltered(t *testing.T) {
	// Every token is structural (no argument values at all)
	resp := &ModelResponse{
		Tokens:   []string{`{"name":"x","arguments":{}}`},
		Logprobs: []float64{-0.01},
	}
	filterToolCallArgTokens(resp)
	if resp.FilteredAverageLogprob != 0 {
		t.Errorf("expected zero FilteredAverageLogprob for empty arguments, got %f", resp.FilteredAverageLogprob)
	}
}

// ---------------------------------------------------------------------------
// ApplyTokenFilter – dispatch tests
// ---------------------------------------------------------------------------

func TestApplyTokenFilter_NilResponse(t *testing.T) {
	// Should not panic
	ApplyTokenFilter(nil, "tool_call_args")
}

func TestApplyTokenFilter_EmptyFilter(t *testing.T) {
	resp := &ModelResponse{
		Tokens:   []string{`{"name":"x","arguments":{"a":"b"}}`},
		Logprobs: []float64{-0.1},
	}
	ApplyTokenFilter(resp, "")
	if resp.FilteredAverageLogprob != 0 {
		t.Errorf("expected no filtering with empty filter, got %f", resp.FilteredAverageLogprob)
	}
}

func TestApplyTokenFilter_AllFilter(t *testing.T) {
	resp := &ModelResponse{
		Tokens:   []string{`{"name":"x","arguments":{"a":"b"}}`},
		Logprobs: []float64{-0.1},
	}
	ApplyTokenFilter(resp, "all")
	if resp.FilteredAverageLogprob != 0 {
		t.Errorf("expected no filtering with 'all' filter, got %f", resp.FilteredAverageLogprob)
	}
}

func TestApplyTokenFilter_UnknownFilter(t *testing.T) {
	resp := &ModelResponse{
		Tokens:   []string{`{"name":"x","arguments":{"a":"b"}}`},
		Logprobs: []float64{-0.1},
	}
	ApplyTokenFilter(resp, "unknown_strategy")
	if resp.FilteredAverageLogprob != 0 {
		t.Errorf("expected no filtering with unknown filter, got %f", resp.FilteredAverageLogprob)
	}
}

func TestApplyTokenFilter_NoTokens(t *testing.T) {
	resp := &ModelResponse{
		Logprobs: []float64{-0.1},
	}
	ApplyTokenFilter(resp, "tool_call_args")
	if resp.FilteredAverageLogprob != 0 {
		t.Errorf("expected no filtering with no tokens, got %f", resp.FilteredAverageLogprob)
	}
}

// ---------------------------------------------------------------------------
// Edge cases and regression tests
// ---------------------------------------------------------------------------

func TestClassifyToolCallChars_TrailingGarbage(t *testing.T) {
	input := `{"name":"act","arguments":{"v":"ok"}}some trailing text`
	mask := classifyToolCallChars(input)
	if mask == nil {
		t.Fatal("expected non-nil mask")
	}
	got := semanticChars(input, mask)
	if got != "ok" {
		t.Errorf("semantic chars = %q, want %q", got, "ok")
	}
}

func TestClassifyToolCallChars_LeadingText(t *testing.T) {
	input := `Here is the tool call: {"name":"act","arguments":{"v":"ok"}}`
	mask := classifyToolCallChars(input)
	if mask == nil {
		t.Fatal("expected non-nil mask")
	}
	got := semanticChars(input, mask)
	if got != "ok" {
		t.Errorf("semantic chars = %q, want %q", got, "ok")
	}
}

func TestClassifyToolCallChars_SpecialCharsInValue(t *testing.T) {
	input := `{"name":"cmd","arguments":{"path":"/usr/bin/ls -la","flag":"--color=auto"}}`
	mask := classifyToolCallChars(input)
	if mask == nil {
		t.Fatal("expected non-nil mask")
	}
	got := semanticChars(input, mask)
	if got != "/usr/bin/ls -la--color=auto" {
		t.Errorf("semantic chars = %q, want %q", got, "/usr/bin/ls -la--color=auto")
	}
}

func TestClassifyToolCallChars_ZeroValue(t *testing.T) {
	input := `{"name":"act","arguments":{"x":0,"y":0}}`
	mask := classifyToolCallChars(input)
	if mask == nil {
		t.Fatal("expected non-nil mask")
	}
	got := semanticChars(input, mask)
	if got != "00" {
		t.Errorf("semantic chars = %q, want %q", got, "00")
	}
}

func TestClassifyToolCallChars_SingleCharTokens(t *testing.T) {
	// Simulates a pathological tokenizer that splits every character
	input := `{"name":"a","arguments":{"k":"v"}}`
	mask := classifyToolCallChars(input)
	if mask == nil {
		t.Fatal("expected non-nil mask")
	}
	got := semanticChars(input, mask)
	if got != "v" {
		t.Errorf("semantic chars = %q, want %q", got, "v")
	}
}

func TestFilterToolCallArgTokens_SingleCharTokenization(t *testing.T) {
	// Each character is its own token — worst-case tokenization
	input := `{"name":"a","arguments":{"k":"v"}}`
	tokens := make([]string, len(input))
	logprobs := make([]float64, len(input))
	for i, c := range input {
		tokens[i] = string(c)
		logprobs[i] = -0.01
	}
	// Make the 'v' token have a distinctive logprob
	vIdx := strings.LastIndex(input, "v")
	logprobs[vIdx] = -0.99

	resp := &ModelResponse{
		Tokens:   tokens,
		Logprobs: logprobs,
	}
	filterToolCallArgTokens(resp)

	if resp.FilteredAverageLogprob == 0 {
		t.Fatal("expected non-zero FilteredAverageLogprob")
	}
	// Only "v" should be semantic → filtered avg should equal its logprob
	if abs(resp.FilteredAverageLogprob-(-0.99)) > 0.001 {
		t.Errorf("FilteredAverageLogprob = %.4f, want -0.99", resp.FilteredAverageLogprob)
	}
}

func TestFilterToolCallArgTokens_MultiCharCoordinateToken(t *testing.T) {
	// Token spans across structural and semantic chars (e.g., ":[499")
	resp := &ModelResponse{
		Tokens:   []string{`{"name":"a","arguments":{"action":"click","coordinate":`, `[499`, `,161`, `]}}`},
		Logprobs: []float64{-0.01, -0.50, -0.60, -0.01},
	}
	filterToolCallArgTokens(resp)

	if resp.FilteredAverageLogprob == 0 {
		t.Fatal("expected non-zero FilteredAverageLogprob")
	}
	// Tokens "[499" and ",161" both contain semantic chars (digits)
	// "click" is merged into the big first token which also contains it
	// Let me reconsider: first token has "click" inside it → it IS semantic
	// So semantic tokens: idx 0 (contains "click"), idx 1 (contains "499"), idx 2 (contains "161")
	expectedAvg := (-0.01 + -0.50 + -0.60) / 3.0
	if abs(resp.FilteredAverageLogprob-expectedAvg) > 0.001 {
		t.Errorf("FilteredAverageLogprob = %.4f, want %.4f", resp.FilteredAverageLogprob, expectedAvg)
	}
}

// ---------------------------------------------------------------------------
// Benchmark
// ---------------------------------------------------------------------------

func BenchmarkClassifyToolCallChars(b *testing.B) {
	input := `{"name":"computer_action","arguments":{"action":"click","coordinate":[499,161]}}`
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		classifyToolCallChars(input)
	}
}

func BenchmarkClassifyToolCallChars_Large(b *testing.B) {
	// Simulate a larger tool call with many arguments
	input := fmt.Sprintf(`{"name":"complex","arguments":{"action":"fill_form","fields":[%s]}}`,
		strings.Repeat(`"field_value",`, 99)+`"last"`)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		classifyToolCallChars(input)
	}
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
