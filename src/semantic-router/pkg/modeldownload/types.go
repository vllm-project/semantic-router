package modeldownload

// ModelSpec represents a model to be downloaded
type ModelSpec struct {
	// Local path where the model should be stored (e.g., "models/mom-embedding-light")
	LocalPath string
	// HuggingFace repository ID (e.g., "sentence-transformers/all-MiniLM-L12-v2")
	RepoID string
	// Git revision (commit hash, tag, or branch). Empty leaves the revision
	// unspecified for local reuse; a download uses HuggingFace's default branch.
	Revision string
	// Required files to verify model completeness
	RequiredFiles []string
	// Each group requires at least one matching file, allowing native sharded
	// weights and provider-supported ONNX layouts.
	RequiredFileGroups [][]string
	// FilesOnly is used for a separate registered mapping or graph artifact.
	FilesOnly bool
	// CheckONNX verifies declared external tensor files for downloaded graphs.
	CheckONNX bool
	// Strict protects explicitly selected deployments and their companion files:
	// populated directories require matching immutable revision metadata before
	// any download; failures cannot degrade to an older or unavailable artifact.
	Strict bool
	// Glob patterns passed to `hf download --exclude` so artifacts the configured
	// runtime never loads are skipped. Empty means the full snapshot is fetched.
	ExcludePatterns []string
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
