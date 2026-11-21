package llmprotocol

// StructuredOutputWithTools reports a request that asks for JSON output on a
// turn whose tools the model may call. Model servers such as vLLM apply the
// output format to the whole completion, so the model could not call a tool
// while the format is set; the OpenAI APIs hold only a final answer without
// tool calls to it.
func StructuredOutputWithTools(request Request) bool {
	switch request.OutputFormat.Kind {
	case OutputJSONObject, OutputJSONSchema:
	default:
		return false
	}
	return len(request.Tools) > 0 && request.ToolChoice.Mode != ToolChoiceNone
}

// FinalAnswer reports an answer that ends its turn on its own without
// calling a tool: no tool call, refusal or generated image, and a stop at the
// end of the turn, at a stop sequence or for no stated reason. An answer cut
// at the output limit, filtered, paused or failed is not one.
func FinalAnswer(response Response) bool {
	if response.Error != nil || len(response.Alternatives) > 0 {
		return false
	}
	switch response.StopReason {
	case StopEndTurn, StopSequence, StopUnknown, "":
	default:
		return false
	}
	for _, item := range response.Output {
		for _, content := range item.Content {
			switch content.Kind {
			case ContentToolCall, ContentRefusal, ContentGeneratedImage:
				return false
			}
		}
	}
	return true
}

// ConstrainedAnswerRequest is the request that asks again for a turn's final
// answer held to its output format: the same request with no tool callable,
// for a buffered answer.
func ConstrainedAnswerRequest(request Request) Request {
	constrained := request
	constrained.ToolChoice = ToolChoice{Mode: ToolChoiceNone}
	constrained.ParallelToolCalls = nil
	constrained.Stream = false
	constrained.StreamOptions = StreamOptions{}
	constrained.Generation++
	return constrained
}
