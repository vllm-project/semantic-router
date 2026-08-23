package config

import (
	"fmt"
	"reflect"
	"strings"
)

var mlSelectionAlgorithmTypes = map[string]struct{}{
	DecisionAlgorithmKNN:    {},
	DecisionAlgorithmKMeans: {},
	DecisionAlgorithmSVM:    {},
	DecisionAlgorithmMLP:    {},
}

// IsMLSelectionAlgorithm reports whether algorithmType selects one of the
// artifact-backed ML selector families.
func IsMLSelectionAlgorithm(algorithmType string) bool {
	_, ok := mlSelectionAlgorithmTypes[strings.ToLower(strings.TrimSpace(algorithmType))]
	return ok
}

// ValidateMLSelectionAlgorithmConfig validates one Decision-local ML selector
// contract. A Decision configures exactly the family named by algorithm.type;
// shared artifact and embedding settings remain inside the same algorithm.ml
// block so a Recipe is self-contained.
func ValidateMLSelectionAlgorithmConfig(algorithmType string, cfg *MLSelectionConfig) error {
	algorithmType = strings.ToLower(strings.TrimSpace(algorithmType))
	if !IsMLSelectionAlgorithm(algorithmType) {
		return fmt.Errorf("algorithm.type=%s is not an ML selector", algorithmType)
	}
	if cfg == nil {
		return fmt.Errorf("algorithm.type=%s requires algorithm.ml configuration", algorithmType)
	}
	if cfg.ModelsPath != strings.TrimSpace(cfg.ModelsPath) {
		return fmt.Errorf("models_path must not contain surrounding whitespace")
	}
	if cfg.EmbeddingDim < 0 {
		return fmt.Errorf("embedding_dim must be positive when configured")
	}

	configuredFamilies := make([]string, 0, 4)
	if cfg.KNN != nil {
		configuredFamilies = append(configuredFamilies, DecisionAlgorithmKNN)
	}
	if cfg.KMeans != nil {
		configuredFamilies = append(configuredFamilies, DecisionAlgorithmKMeans)
	}
	if cfg.SVM != nil {
		configuredFamilies = append(configuredFamilies, DecisionAlgorithmSVM)
	}
	if cfg.MLP != nil {
		configuredFamilies = append(configuredFamilies, DecisionAlgorithmMLP)
	}
	if len(configuredFamilies) != 1 || configuredFamilies[0] != algorithmType {
		if len(configuredFamilies) == 0 {
			return fmt.Errorf("algorithm.type=%s requires algorithm.ml.%s configuration", algorithmType, algorithmType)
		}
		return fmt.Errorf(
			"algorithm.type=%s requires only algorithm.ml.%s configuration; found %s",
			algorithmType,
			algorithmType,
			strings.Join(configuredFamilies, ", "),
		)
	}

	return validateMLSelectionFamilyValues(algorithmType, cfg)
}

func validateMLSelectionFamilyValues(algorithmType string, cfg *MLSelectionConfig) error {
	switch algorithmType {
	case DecisionAlgorithmKNN:
		if cfg.KNN.K < 0 {
			return fmt.Errorf("knn.k must be positive when configured")
		}
		return validateMLPretrainedPath("knn", cfg.KNN.PretrainedPath)
	case DecisionAlgorithmKMeans:
		if cfg.KMeans.NumClusters < 0 {
			return fmt.Errorf("kmeans.num_clusters must be positive when configured")
		}
		if cfg.KMeans.EfficiencyWeight < 0 || cfg.KMeans.EfficiencyWeight > 1 {
			return fmt.Errorf("kmeans.efficiency_weight must be between 0 and 1")
		}
		return validateMLPretrainedPath("kmeans", cfg.KMeans.PretrainedPath)
	case DecisionAlgorithmSVM:
		kernel := strings.ToLower(strings.TrimSpace(cfg.SVM.Kernel))
		if kernel != "" && kernel != "linear" && kernel != "rbf" && kernel != "gaussian" {
			return fmt.Errorf("svm.kernel must be linear, rbf, or gaussian")
		}
		if cfg.SVM.Gamma < 0 {
			return fmt.Errorf("svm.gamma must not be negative")
		}
		return validateMLPretrainedPath("svm", cfg.SVM.PretrainedPath)
	case DecisionAlgorithmMLP:
		device := strings.ToLower(strings.TrimSpace(cfg.MLP.Device))
		if device != "" && device != "cpu" && device != "cuda" && device != "metal" {
			return fmt.Errorf("mlp.device must be cpu, cuda, or metal")
		}
		return validateMLPretrainedPath("mlp", cfg.MLP.PretrainedPath)
	default:
		return fmt.Errorf("unsupported ML selector %q", algorithmType)
	}
}

func validateMLPretrainedPath(family string, path string) error {
	if path != strings.TrimSpace(path) {
		return fmt.Errorf("%s.pretrained_path must not contain surrounding whitespace", family)
	}
	return nil
}

// MLSelectionConfigForRoutingProfile returns the aggregate selector registry
// configuration for exactly one routing profile. Different ML families may
// coexist when they agree on shared settings. Repeated configuration of the
// same family must be identical, which prevents one Recipe from giving a
// mutable registry contradictory initialization inputs.
func MLSelectionConfigForRoutingProfile(cfg *RouterConfig) (*MLSelectionConfig, error) {
	if cfg == nil {
		return nil, nil
	}

	var aggregate *MLSelectionConfig
	sharedOwner := ""
	familyOwners := make(map[string]string, 4)
	for _, decision := range cfg.Decisions {
		if decision.Algorithm == nil {
			continue
		}
		algorithmType := strings.ToLower(strings.TrimSpace(decision.Algorithm.Type))
		if !IsMLSelectionAlgorithm(algorithmType) {
			if decision.Algorithm.ML != nil {
				return nil, fmt.Errorf(
					"decision %q: algorithm.ml requires an ML algorithm.type",
					decision.Name,
				)
			}
			continue
		}
		if err := ValidateMLSelectionAlgorithmConfig(algorithmType, decision.Algorithm.ML); err != nil {
			return nil, fmt.Errorf("decision %q: %w", decision.Name, err)
		}
		if aggregate == nil {
			aggregate = &MLSelectionConfig{
				ModelsPath:   decision.Algorithm.ML.ModelsPath,
				EmbeddingDim: decision.Algorithm.ML.EmbeddingDim,
			}
			sharedOwner = decision.Name
		} else if aggregate.ModelsPath != decision.Algorithm.ML.ModelsPath ||
			aggregate.EmbeddingDim != decision.Algorithm.ML.EmbeddingDim {
			return nil, fmt.Errorf(
				"decisions %q and %q configure conflicting algorithm.ml shared settings",
				sharedOwner,
				decision.Name,
			)
		}
		if err := mergeMLSelectionFamily(aggregate, algorithmType, decision.Name, decision.Algorithm.ML, familyOwners); err != nil {
			return nil, err
		}
	}
	return aggregate, nil
}

func mergeMLSelectionFamily(
	aggregate *MLSelectionConfig,
	family string,
	decisionName string,
	configured *MLSelectionConfig,
	owners map[string]string,
) error {
	conflicts := false
	switch family {
	case DecisionAlgorithmKNN:
		if aggregate.KNN == nil {
			aggregate.KNN = cloneMLKNNConfig(configured.KNN)
			owners[family] = decisionName
			return nil
		}
		conflicts = !reflect.DeepEqual(aggregate.KNN, configured.KNN)
	case DecisionAlgorithmKMeans:
		if aggregate.KMeans == nil {
			aggregate.KMeans = cloneMLKMeansConfig(configured.KMeans)
			owners[family] = decisionName
			return nil
		}
		conflicts = !reflect.DeepEqual(aggregate.KMeans, configured.KMeans)
	case DecisionAlgorithmSVM:
		if aggregate.SVM == nil {
			aggregate.SVM = cloneMLSVMConfig(configured.SVM)
			owners[family] = decisionName
			return nil
		}
		conflicts = !reflect.DeepEqual(aggregate.SVM, configured.SVM)
	case DecisionAlgorithmMLP:
		if aggregate.MLP == nil {
			aggregate.MLP = cloneMLMLPConfig(configured.MLP)
			owners[family] = decisionName
			return nil
		}
		conflicts = !reflect.DeepEqual(aggregate.MLP, configured.MLP)
	}
	if conflicts {
		return fmt.Errorf(
			"decisions %q and %q configure conflicting algorithm.ml.%s settings",
			owners[family],
			decisionName,
			family,
		)
	}
	return nil
}

func cloneMLKNNConfig(cfg *MLKNNConfig) *MLKNNConfig {
	if cfg == nil {
		return nil
	}
	copy := *cfg
	return &copy
}

func cloneMLKMeansConfig(cfg *MLKMeansConfig) *MLKMeansConfig {
	if cfg == nil {
		return nil
	}
	copy := *cfg
	return &copy
}

func cloneMLSVMConfig(cfg *MLSVMConfig) *MLSVMConfig {
	if cfg == nil {
		return nil
	}
	copy := *cfg
	return &copy
}

func cloneMLMLPConfig(cfg *MLMLPConfig) *MLMLPConfig {
	if cfg == nil {
		return nil
	}
	copy := *cfg
	return &copy
}

func validateRoutingProfileMLSelectionConfig(cfg *RouterConfig) error {
	_, err := MLSelectionConfigForRoutingProfile(cfg)
	return err
}

// HasReachableMLSelection reports whether any request-reachable Recipe uses
// an artifact-backed ML selector. Runtime model initialization uses this to
// enable batched embeddings without consulting a global selector block.
func (c *RouterConfig) HasReachableMLSelection() bool {
	if c == nil {
		return false
	}
	if c.RoutingScope != "" {
		ml, err := MLSelectionConfigForRoutingProfile(c)
		return err == nil && ml != nil
	}
	reachable := c.ReachableRoutingRecipes()
	for _, recipe := range reachable {
		ml, err := MLSelectionConfigForRoutingProfile(c.ConfigForRecipe(recipe))
		if err == nil && ml != nil {
			return true
		}
	}
	return false
}
