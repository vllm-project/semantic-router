package protocolcodec

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"

func (decoder *responsesStreamDecoder) applyResponsesImageGenerationProgress(
	event *llmprotocol.Event,
	wire responsesEventWire,
) error {
	status := llmprotocol.ImageGenerationGenerating
	switch wire.Type {
	case "response.image_generation_call.in_progress":
		status = llmprotocol.ImageGenerationInProgress
	case "response.image_generation_call.completed":
		status = llmprotocol.ImageGenerationCompleted
	}
	image := &llmprotocol.GeneratedImage{
		Status: status, PartialIndex: wire.PartialImageIndex,
		PartialImage: wire.PartialImageB64, Size: wire.Size, Quality: wire.Quality,
		Background: wire.Background, OutputFormat: wire.OutputFormat,
	}
	event.Type = llmprotocol.EventImageGenerationProgress
	event.GeneratedImage = image
	return nil
}

func (encoder *responsesStreamEncoder) encodeResponsesImageGenerationProgress(
	event llmprotocol.Event,
) ([][]byte, error) {
	if event.GeneratedImage == nil {
		return nil, llmprotocol.NewError(
			llmprotocol.ErrorInternal,
			"image_generation_event_invalid",
			"image generation event is invalid",
			nil,
		)
	}
	frames, key, err := encoder.ensureResponsesOutputStarted(event, responsesOutputImage)
	if err != nil {
		return nil, err
	}
	wire := responsesEventWire{
		Sequence: encoder.nextWireSequence(), ItemID: encoder.outputIDs[key],
		OutputIndex: responsesOutputIndex(encoder.outputIndexes[key]),
	}
	image := event.GeneratedImage
	switch {
	case image.PartialIndex != nil:
		wire.Type = "response.image_generation_call.partial_image"
		wire.PartialImageIndex = image.PartialIndex
		wire.PartialImageB64 = image.PartialImage
		wire.Size, wire.Quality = image.Size, image.Quality
		wire.Background, wire.OutputFormat = image.Background, image.OutputFormat
	case image.Status == llmprotocol.ImageGenerationInProgress:
		wire.Type = "response.image_generation_call.in_progress"
	case image.Status == llmprotocol.ImageGenerationGenerating:
		wire.Type = "response.image_generation_call.generating"
	case image.Status == llmprotocol.ImageGenerationCompleted:
		wire.Type = "response.image_generation_call.completed"
		encoder.imageProgressCompleted[key] = true
	default:
		return nil, llmprotocol.NewError(
			llmprotocol.ErrorUnsupportedFeature,
			"image_generation_stream_status",
			"image generation status has no Responses stream event",
			nil,
		)
	}
	frame, err := encoder.encodeResponsesStreamFrame(wire)
	return append(frames, frame), err
}

func (encoder *responsesStreamEncoder) encodeCompletedResponsesImage(
	event llmprotocol.Event,
	key responsesOutputKey,
) ([][]byte, llmprotocol.Diagnostics, error) {
	if event.Content == nil || event.Content.GeneratedImage == nil {
		return nil, nil, llmprotocol.NewError(
			llmprotocol.ErrorInternal,
			"image_generation_item_invalid",
			"image generation completion is invalid",
			nil,
		)
	}
	image := event.Content.GeneratedImage
	var frames [][]byte
	if image.Status == llmprotocol.ImageGenerationCompleted && !encoder.imageProgressCompleted[key] {
		progress := event
		progress.Type = llmprotocol.EventImageGenerationProgress
		progress.GeneratedImage = &llmprotocol.GeneratedImage{Status: llmprotocol.ImageGenerationCompleted}
		generated, err := encoder.encodeResponsesImageGenerationProgress(progress)
		if err != nil {
			return nil, nil, err
		}
		frames = append(frames, generated...)
	}
	index, id := encoder.outputIndexes[key], encoder.outputIDs[key]
	item := responsesItemWire{
		Type: "image_generation_call", ID: id, Status: string(image.Status),
	}
	if image.Result != nil {
		result := *image.Result
		item.Result = &result
	}
	wire := responsesEventWire{
		Type: "response.output_item.done", Sequence: encoder.nextWireSequence(),
		OutputIndex: responsesOutputIndex(index), Item: marshalResponsesEventItem(item),
	}
	encoder.recordResponsesCompletedOutput(index, wire.Item)
	frame, err := encoder.encodeResponsesStreamFrame(wire)
	return append(frames, frame), nil, err
}
