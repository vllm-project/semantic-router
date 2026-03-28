package mlpipeline

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"gopkg.in/yaml.v3"
)

// GenerateConfig runs Layer 3: generates a deployment-ready YAML config.
func (r *Runner) GenerateConfig(req ConfigRequest) (string, error) {
	job := r.createJob("config")
	jobDir := r.JobDir(job.ID)
	if err := ensureDir(jobDir); err != nil {
		return "", fmt.Errorf("failed to create job dir: %w", err)
	}

	r.mu.Lock()
	job.Status = StatusRunning
	r.mu.Unlock()

	r.sendProgress(job.ID, 10, "Generating config", "Building deployment configuration")

	configMap := buildConfigYAML(req)

	outputPath := filepath.Join(jobDir, "ml-model-selection-values.yaml")
	var buf bytes.Buffer
	enc := yaml.NewEncoder(&buf)
	enc.SetIndent(2)
	if err := enc.Encode(configMap); err != nil {
		r.failJob(job.ID, fmt.Sprintf("failed to marshal config: %v", err))
		return "", err
	}
	enc.Close()

	yamlStr := buf.String()
	lines := strings.Split(yamlStr, "\n")
	var out []string
	decisionCount := 0
	for _, line := range lines {
		if strings.HasPrefix(strings.TrimRight(line, " "), "    - name:") {
			decisionCount++
			if decisionCount > 1 && len(out) > 0 && strings.TrimSpace(out[len(out)-1]) != "" {
				out = append(out, "")
			}
		}
		out = append(out, line)
	}
	finalYAML := strings.Join(out, "\n")

	if err := os.WriteFile(outputPath, []byte(finalYAML), 0o644); err != nil {
		r.failJob(job.ID, fmt.Sprintf("failed to write config: %v", err))
		return "", err
	}

	r.completeJob(job.ID, []string{outputPath})
	r.sendProgress(job.ID, 100, "Completed", "Config generated successfully")

	return job.ID, nil
}

// YAML config structs matching semantic-router values.yaml format.
type yamlConfig struct {
	Config yamlConfigInner `yaml:"config"`
}

type yamlConfigInner struct {
	ModelSelection yamlModelSelection `yaml:"model_selection"`
	Strategy       string             `yaml:"strategy"`
	Decisions      []yamlDecision     `yaml:"decisions"`
}

type yamlModelSelection struct {
	Enabled bool   `yaml:"enabled"`
	ML      yamlML `yaml:"ml"`
}

type yamlML struct {
	ModelsPath   string      `yaml:"models_path"`
	EmbeddingDim int         `yaml:"embedding_dim"`
	KNN          *yamlKNN    `yaml:"knn,omitempty"`
	KMeans       *yamlKMeans `yaml:"kmeans,omitempty"`
	SVM          *yamlSVM    `yaml:"svm,omitempty"`
	MLP          *yamlMLP    `yaml:"mlp,omitempty"`
}

type yamlKNN struct {
	K              int    `yaml:"k"`
	PretrainedPath string `yaml:"pretrained_path"`
}

type yamlKMeans struct {
	NumClusters    int    `yaml:"num_clusters"`
	PretrainedPath string `yaml:"pretrained_path"`
}

type yamlSVM struct {
	Kernel         string  `yaml:"kernel"`
	Gamma          float64 `yaml:"gamma"`
	PretrainedPath string  `yaml:"pretrained_path"`
}

type yamlMLP struct {
	Device         string `yaml:"device"`
	PretrainedPath string `yaml:"pretrained_path"`
}

type yamlDecision struct {
	Name      string         `yaml:"name"`
	Priority  int            `yaml:"priority"`
	Rules     yamlRules      `yaml:"rules"`
	Algorithm yamlAlgorithm  `yaml:"algorithm"`
	ModelRefs []yamlModelRef `yaml:"modelRefs"`
}

type yamlRules struct {
	Operator   string          `yaml:"operator"`
	Conditions []yamlCondition `yaml:"conditions"`
}

type yamlCondition struct {
	Type string `yaml:"type"`
	Name string `yaml:"name"`
}

type yamlAlgorithm struct {
	Type string `yaml:"type"`
}

type yamlModelRef struct {
	Model        string `yaml:"model"`
	UseReasoning bool   `yaml:"use_reasoning"`
}

// buildConfigYAML creates the config structure matching the semantic-router values.yaml format.
func buildConfigYAML(req ConfigRequest) yamlConfig {
	modelsPath := req.ModelsPath
	if modelsPath == "" {
		modelsPath = "/data/ml-pipeline/ml-train"
	}

	ml := yamlML{
		ModelsPath:   modelsPath,
		EmbeddingDim: 1024,
	}

	algSet := map[string]bool{}
	for _, d := range req.Decisions {
		algSet[d.Algorithm] = true
	}
	if algSet["knn"] {
		ml.KNN = &yamlKNN{
			K:              5,
			PretrainedPath: filepath.Join(modelsPath, "knn_model.json"),
		}
	}
	if algSet["kmeans"] {
		ml.KMeans = &yamlKMeans{
			NumClusters:    8,
			PretrainedPath: filepath.Join(modelsPath, "kmeans_model.json"),
		}
	}
	if algSet["svm"] {
		ml.SVM = &yamlSVM{
			Kernel:         "rbf",
			Gamma:          1.0,
			PretrainedPath: filepath.Join(modelsPath, "svm_model.json"),
		}
	}
	if algSet["mlp"] {
		mlpDevice := req.Device
		if mlpDevice == "" {
			mlpDevice = "cpu"
		}
		ml.MLP = &yamlMLP{
			Device:         mlpDevice,
			PretrainedPath: filepath.Join(modelsPath, "mlp_model.json"),
		}
	}

	decisions := []yamlDecision{}
	for _, d := range req.Decisions {
		conditions := []yamlCondition{}
		for _, dom := range d.Domains {
			conditions = append(conditions, yamlCondition{
				Type: "domain",
				Name: dom,
			})
		}

		modelRefs := []yamlModelRef{}
		for _, mn := range d.ModelNames {
			modelRefs = append(modelRefs, yamlModelRef{
				Model:        mn,
				UseReasoning: false,
			})
		}

		decisions = append(decisions, yamlDecision{
			Name:     d.Name,
			Priority: d.Priority,
			Rules: yamlRules{
				Operator:   "OR",
				Conditions: conditions,
			},
			Algorithm: yamlAlgorithm{Type: d.Algorithm},
			ModelRefs: modelRefs,
		})
	}

	return yamlConfig{
		Config: yamlConfigInner{
			ModelSelection: yamlModelSelection{
				Enabled: true,
				ML:      ml,
			},
			Strategy:  "priority",
			Decisions: decisions,
		},
	}
}
