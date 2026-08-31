package llmprotocol

// ImageGenerationOptions is the protocol-neutral configuration for a hosted
// image-generation tool. Wire codecs own provider spellings; Router plugins may
// consume these semantics without inspecting provider JSON.
type ImageGenerationOptions struct {
	Model             string
	Quality           string
	Size              string
	OutputFormat      string
	OutputCompression *int64
	Moderation        string
	Background        string
	InputFidelity     string
	InputImageMask    *ImageGenerationMask
	PartialImages     *int64
	Action            string
}

type ImageGenerationMask struct {
	EncodedImage string
	FileID       string
}

type ImageGenerationStatus string

const (
	ImageGenerationInProgress ImageGenerationStatus = "in_progress"
	ImageGenerationGenerating ImageGenerationStatus = "generating"
	ImageGenerationCompleted  ImageGenerationStatus = "completed"
	ImageGenerationFailed     ImageGenerationStatus = "failed"
)

// GeneratedImage preserves both terminal output items and streaming progress.
// Result is a pointer because the Responses contract distinguishes a null
// result from an explicitly present empty base64 payload. PartialImage is
// populated only for a partial-image event.
type GeneratedImage struct {
	Status       ImageGenerationStatus
	Result       *string
	PartialIndex *int64
	PartialImage string
	Size         string
	Quality      string
	Background   string
	OutputFormat string
}
