package config

// ModelRegistryInfo represents canonical metadata about a router-registered model.
// It uses the local registry as a baseline and may be overlaid with Hugging Face
// model-card/config metadata when available.
type ModelRegistryInfo struct {
	LocalPath           string   `json:"local_path,omitempty"`
	RepoID              string   `json:"repo_id,omitempty"`
	Purpose             string   `json:"purpose,omitempty"`
	Description         string   `json:"description,omitempty"`
	ParameterSize       string   `json:"parameter_size,omitempty"`
	EmbeddingDim        int      `json:"embedding_dim,omitempty"`
	MaxContextLength    int      `json:"max_context_length,omitempty"`
	BaseModelMaxContext int      `json:"base_model_max_context,omitempty"`
	UsesLoRA            bool     `json:"uses_lora,omitempty"`
	NumClasses          int      `json:"num_classes,omitempty"`
	Tags                []string `json:"tags,omitempty"`
	ModelCardURL        string   `json:"model_card_url,omitempty"`
	PipelineTag         string   `json:"pipeline_tag,omitempty"`
	BaseModel           string   `json:"base_model,omitempty"`
	License             string   `json:"license,omitempty"`
	Languages           []string `json:"languages,omitempty"`
	Datasets            []string `json:"datasets,omitempty"`
}

func (m ModelSpec) RegistryInfo() ModelRegistryInfo {
	return ModelRegistryInfo{
		LocalPath:           m.LocalPath,
		RepoID:              m.RepoID,
		Purpose:             string(m.Purpose),
		Description:         m.Description,
		ParameterSize:       m.ParameterSize,
		EmbeddingDim:        m.EmbeddingDim,
		MaxContextLength:    m.MaxContextLength,
		BaseModelMaxContext: m.BaseModelMaxContext,
		UsesLoRA:            m.UsesLoRA,
		NumClasses:          m.NumClasses,
		Tags:                cloneStrings(m.Tags),
	}
}

func cloneStrings(values []string) []string {
	if len(values) == 0 {
		return nil
	}

	cloned := make([]string, len(values))
	copy(cloned, values)
	return cloned
}
