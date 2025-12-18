package modeldownload

// ModelSpec represents a model to be downloaded
type ModelSpec struct {
	// Local path where the model should be stored (e.g., "models/all-MiniLM-L12-v2")
	LocalPath string
	// HuggingFace repository ID (e.g., "sentence-transformers/all-MiniLM-L12-v2")
	RepoID string
	// Git revision (commit hash, tag, or branch). Defaults to "main"
	Revision string
	// Required files to verify model completeness
	RequiredFiles []string
}

// DownloadConfig contains configuration for model downloading
type DownloadConfig struct {
	// HuggingFace endpoint URL
	HFEndpoint string
	// HuggingFace access token for private repositories
	HFToken string
	// Cache directory for HuggingFace downloads
	HFHome string
}

// MoMRegistry maps local paths to HuggingFace repo IDs
type MoMRegistry map[string]string
