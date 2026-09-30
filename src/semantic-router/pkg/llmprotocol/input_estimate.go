package llmprotocol

import "math"

// These provider-neutral reserves are estimates, not tokenizer bounds. Media
// preprocessing can differ by provider; callers must not present them as exact.
const (
	InputTextBytesPerToken           = 4
	InputImageTokens                 = 8 * 1024
	InputMessageFramingTokens        = 4
	InputToolCallFramingTokens       = 8
	InputToolDefinitionFramingTokens = 8
)

// InputEstimate excludes generation limits. Consumers add the effective output
// reserve exactly once after resolving the selected model's known limits.
type InputEstimate struct {
	Tokens          int
	TextBytes       int
	StructuredBytes int
	EquivalentBytes int
	HasNonText      bool
}

// EstimateInput walks neutral content, including retained history and tool
// results. It performs no tokenization, media fetching, or model inference.
func EstimateInput(request *Request) InputEstimate {
	if request == nil {
		return InputEstimate{}
	}
	var estimate InputEstimate
	framing := saturatedMultiply(len(request.Messages)+len(request.Instructions), InputMessageFramingTokens)
	images := 0
	var consume func([]Content)
	consume = func(contents []Content) {
		for _, content := range contents {
			switch content.Kind {
			case ContentText, ContentRefusal, ContentReasoning:
				estimate.TextBytes = SaturatingTokenSum(estimate.TextBytes, len(content.Text))
			case ContentImage:
				estimate.HasNonText = true
				images = SaturatingTokenSum(images, 1)
			case ContentAudio, ContentVideo, ContentFile:
				estimate.HasNonText = true
				estimate.StructuredBytes = SaturatingTokenSum(estimate.StructuredBytes, len(content.URL), len(content.Data), len(content.FileID))
			case ContentToolCall:
				framing = SaturatingTokenSum(framing, InputToolCallFramingTokens)
				if content.ToolCall != nil {
					estimate.StructuredBytes = SaturatingTokenSum(estimate.StructuredBytes, len(content.ToolCall.Name), len(content.ToolCall.Arguments))
				}
			case ContentToolResult:
				estimate.HasNonText = true
				if content.ToolResult != nil {
					consume(content.ToolResult.Content)
				}
			}
		}
	}
	for _, instruction := range request.Instructions {
		consume(instruction.Content)
	}
	for _, message := range request.Messages {
		consume(message.Content)
	}
	for _, tool := range request.Tools {
		estimate.TextBytes = SaturatingTokenSum(estimate.TextBytes, len(tool.Name), len(tool.Description))
		estimate.StructuredBytes = SaturatingTokenSum(estimate.StructuredBytes, len(tool.InputSchema))
		if tool.CustomFormat != nil {
			estimate.StructuredBytes = SaturatingTokenSum(estimate.StructuredBytes, len(tool.CustomFormat.Definition))
		}
		framing = SaturatingTokenSum(framing, InputToolDefinitionFramingTokens)
	}
	// A constrained output schema is prompt-bearing structure, even though it
	// is carried outside messages on the wire. Reserve its bytes consistently
	// with tool schemas; this is conservative accounting, not tokenization.
	estimate.StructuredBytes = SaturatingTokenSum(estimate.StructuredBytes, len(request.OutputFormat.Schema), len(request.OutputFormat.Name), len(request.OutputFormat.Description))
	textTokens := estimate.TextBytes / InputTextBytesPerToken
	if estimate.TextBytes%InputTextBytesPerToken != 0 {
		textTokens++
	}
	estimate.Tokens = SaturatingTokenSum(textTokens, estimate.StructuredBytes, saturatedMultiply(images, InputImageTokens), framing)
	estimate.EquivalentBytes = saturatedMultiply(estimate.Tokens, InputTextBytesPerToken)
	return estimate
}

// SaturatingTokenSum avoids overflowing admission calculations.
func SaturatingTokenSum(values ...int) int {
	total := 0
	for _, value := range values {
		if value <= 0 {
			continue
		}
		if total > math.MaxInt-value {
			return math.MaxInt
		}
		total += value
	}
	return total
}

func saturatedMultiply(value, factor int) int {
	if value <= 0 || factor <= 0 {
		return 0
	}
	if value > math.MaxInt/factor {
		return math.MaxInt
	}
	return value * factor
}
