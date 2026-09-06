package protocolcodec

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"

func (state *streamState) applyImageGenerationProgress(event llmprotocol.Event) (llmprotocol.Event, error) {
	if !state.items[event.ItemIndex] || state.completedItems[event.ItemIndex] {
		return llmprotocol.Event{}, llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"invalid_item_lifecycle",
			"upstream image generation event does not reference an active output item",
			nil,
		)
	}
	if event.ItemID != "" && event.ItemID != state.itemIDs[event.ItemIndex] {
		return llmprotocol.Event{}, llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"stream_item_id_mismatch",
			"upstream image generation event changed its output item ID",
			nil,
		)
	}
	event.ItemID = state.itemIDs[event.ItemIndex]
	if state.itemKinds[event.ItemIndex] != llmprotocol.ContentGeneratedImage {
		return llmprotocol.Event{}, llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"stream_item_kind_mismatch",
			"upstream image generation event does not target an image-generation item",
			nil,
		)
	}
	if err := llmprotocol.ValidateGeneratedImage(event.GeneratedImage, state.policy.Limits); err != nil {
		return llmprotocol.Event{}, upstreamSemanticValidationError(err)
	}
	if err := state.validateImageGenerationProgress(event.ItemIndex, event.GeneratedImage); err != nil {
		return llmprotocol.Event{}, err
	}
	return event, nil
}

func (state *streamState) validateImageGenerationProgress(
	index int,
	image *llmprotocol.GeneratedImage,
) error {
	if image.Result != nil {
		return llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"stream_image_generation_result",
			"image generation progress cannot contain a final result",
			nil,
		)
	}
	rank := imageGenerationProgressRank(image.Status)
	if rank == 0 {
		return llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"stream_image_generation_status",
			"image generation progress status is invalid",
			nil,
		)
	}
	if previous := state.imageProgressRank[index]; rank < previous {
		return llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"stream_image_generation_order",
			"image generation progress moved backwards",
			nil,
		)
	}
	seen := state.imageProgressSeen[index]
	if seen == nil {
		seen = make(map[llmprotocol.ImageGenerationStatus]bool)
		state.imageProgressSeen[index] = seen
	}
	if image.PartialIndex == nil && seen[image.Status] {
		return llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"duplicate_stream_image_generation_event",
			"image generation progress event was emitted more than once",
			nil,
		)
	}
	if err := state.validatePartialImageProgress(index, image); err != nil {
		return err
	}
	if image.PartialIndex == nil {
		seen[image.Status] = true
	}
	state.imageProgressRank[index] = rank
	return nil
}

func imageGenerationProgressRank(status llmprotocol.ImageGenerationStatus) int {
	switch status {
	case llmprotocol.ImageGenerationInProgress:
		return 1
	case llmprotocol.ImageGenerationGenerating:
		return 2
	case llmprotocol.ImageGenerationCompleted:
		return 3
	default:
		return 0
	}
}

func (state *streamState) validatePartialImageProgress(
	index int,
	image *llmprotocol.GeneratedImage,
) error {
	if image.PartialIndex == nil {
		if image.PartialImage != "" || image.Size != "" || image.Quality != "" ||
			image.Background != "" || image.OutputFormat != "" {
			return llmprotocol.NewError(
				llmprotocol.ErrorUpstreamUnavailable,
				"stream_partial_image_target",
				"partial image metadata requires a partial image index",
				nil,
			)
		}
		return nil
	}
	if image.Status != llmprotocol.ImageGenerationGenerating {
		return llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"stream_partial_image_status",
			"partial image progress must use generating status",
			nil,
		)
	}
	if *image.PartialIndex != state.nextPartialImageIndex[index] {
		return llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"stream_partial_image_index_order",
			"partial image indexes must be contiguous from zero",
			nil,
		)
	}
	state.nextPartialImageIndex[index]++
	return nil
}

func validateStartedGeneratedImage(image *llmprotocol.GeneratedImage) error {
	if image.Status != llmprotocol.ImageGenerationInProgress || image.Result != nil ||
		image.PartialIndex != nil || image.PartialImage != "" || image.Size != "" ||
		image.Quality != "" || image.Background != "" || image.OutputFormat != "" {
		return llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"stream_image_generation_start",
			"image generation output item must start in progress without result data",
			nil,
		)
	}
	return nil
}

func (state *streamState) validateCompletedGeneratedImage(
	index int,
	image *llmprotocol.GeneratedImage,
) error {
	if image.Status != llmprotocol.ImageGenerationCompleted && image.Status != llmprotocol.ImageGenerationFailed {
		return llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"stream_image_generation_completion",
			"image generation output item must complete with completed or failed status",
			nil,
		)
	}
	if image.PartialIndex != nil || image.PartialImage != "" || image.Size != "" || image.Quality != "" ||
		image.Background != "" || image.OutputFormat != "" {
		return llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"stream_image_generation_completion",
			"image generation output item cannot contain progress-only fields",
			nil,
		)
	}
	if state.imageProgressRank[index] == imageGenerationProgressRank(llmprotocol.ImageGenerationCompleted) &&
		image.Status != llmprotocol.ImageGenerationCompleted {
		return llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"stream_image_generation_completion_mismatch",
			"completed image generation progress cannot finish as failed",
			nil,
		)
	}
	return nil
}
