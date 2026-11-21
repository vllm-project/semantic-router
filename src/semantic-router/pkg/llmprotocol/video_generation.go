package llmprotocol

import "time"

// VideoGenerationOptions is the protocol-neutral configuration for a hosted
// video-generation job.
type VideoGenerationOptions struct {
	Prompt    string
	Model     string
	Seconds   string
	Size      string
	Width     int64
	Height    int64
	FPS       float64
	NumFrames int64
	Quality   string
}

type VideoGenerationStatus string

const (
	VideoGenerationQueued     VideoGenerationStatus = "queued"
	VideoGenerationInProgress VideoGenerationStatus = "in_progress"
	VideoGenerationCompleted  VideoGenerationStatus = "completed"
	VideoGenerationFailed     VideoGenerationStatus = "failed"
)

// GeneratedVideo is one hosted video job or one finished artifact. ID is the
// provider's id: routing never reads it, and the codec never invents one.
type GeneratedVideo struct {
	ID          string
	Status      VideoGenerationStatus
	Progress    *int64
	Seconds     string
	Size        string
	CreatedAt   *time.Time
	CompletedAt *time.Time
	ExpiresAt   *time.Time
	Deleted     bool
	Err         *ProtocolError
}
