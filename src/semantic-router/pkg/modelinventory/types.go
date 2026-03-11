package modelinventory

import routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"

// ModelsInfoResponse represents the response for models info endpoint.
type ModelsInfoResponse struct {
	Models  []ModelInfo       `json:"models"`
	Summary ModelsInfoSummary `json:"summary"`
	System  SystemInfo        `json:"system"`
}

// ModelsInfoSummary represents aggregate runtime information for loaded models.
type ModelsInfoSummary struct {
	Ready            bool     `json:"ready"`
	Phase            string   `json:"phase,omitempty"`
	Message          string   `json:"message,omitempty"`
	DownloadingModel string   `json:"downloading_model,omitempty"`
	PendingModels    []string `json:"pending_models,omitempty"`
	LoadedModels     int      `json:"loaded_models"`
	TotalModels      int      `json:"total_models"`
	UpdatedAt        string   `json:"updated_at,omitempty"`
}

// ModelInfo represents information about a loaded model.
type ModelInfo struct {
	Name              string                          `json:"name"`
	Type              string                          `json:"type"`
	Loaded            bool                            `json:"loaded"`
	State             string                          `json:"state,omitempty"`
	ModelPath         string                          `json:"model_path,omitempty"`
	ResolvedModelPath string                          `json:"resolved_model_path,omitempty"`
	Categories        []string                        `json:"categories,omitempty"`
	Metadata          map[string]string               `json:"metadata,omitempty"`
	Registry          *routerconfig.ModelRegistryInfo `json:"registry,omitempty"`
	LoadTime          string                          `json:"load_time,omitempty"`
	MemoryUsage       string                          `json:"memory_usage,omitempty"`
}

// SystemInfo represents system information.
type SystemInfo struct {
	GoVersion    string `json:"go_version"`
	Architecture string `json:"architecture"`
	OS           string `json:"os"`
	MemoryUsage  string `json:"memory_usage"`
	GPUAvailable bool   `json:"gpu_available"`
}
