package llmprotocol

// AudioOutputOptions is the protocol-neutral request for an audio artifact.
type AudioOutputOptions struct {
	Voice  string
	Format string
}

// GeneratedAudio is the artifact a Chat backend returns on the message.
type GeneratedAudio struct {
	ID         string
	Data       string
	ExpiresAt  int64
	Transcript string
}
