package extproc

import (
	"fmt"
	"log"
	"sync"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tools"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/pii"
)

var (
	initialized bool
	initMutex   sync.Mutex
)

// OpenAIRouter is an Envoy ExtProc server that routes OpenAI API requests
type OpenAIRouter struct {
	Config               *config.RouterConfig
	CategoryDescriptions []string
	Classifier           *classification.Classifier
	PIIChecker           *pii.PolicyChecker
	Cache                cache.CacheBackend
	ToolsDatabase        *tools.ToolsDatabase

	// Map to track pending requests and their unique IDs
	pendingRequests     map[string][]byte
	pendingRequestsLock sync.Mutex
}

// Ensure OpenAIRouter implements the ext_proc calls
var _ ext_proc.ExternalProcessorServer = (*OpenAIRouter)(nil)

// NewOpenAIRouter creates a new OpenAI API router instance
func NewOpenAIRouter(configPath string) (*OpenAIRouter, error) {
	// Always parse fresh config for router construction (supports live reload)
	cfg, err := config.ParseConfigFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load config: %w", err)
	}
	// Update global config reference for packages that rely on config.GetConfig()
	config.ReplaceGlobalConfig(cfg)

	initMutex.Lock()
	defer initMutex.Unlock()

	// Load category mapping if classifier is enabled
	var categoryMapping *classification.CategoryMapping
	if cfg.Classifier.CategoryModel.CategoryMappingPath != "" {
		categoryMapping, err = classification.LoadCategoryMapping(cfg.Classifier.CategoryModel.CategoryMappingPath)
		if err != nil {
			return nil, fmt.Errorf("failed to load category mapping: %w", err)
		}
		log.Printf("Loaded category mapping with %d categories", categoryMapping.GetCategoryCount())
	}

	// Load PII mapping if PII classifier is enabled
	var piiMapping *classification.PIIMapping
	if cfg.Classifier.PIIModel.PIIMappingPath != "" {
		piiMapping, err = classification.LoadPIIMapping(cfg.Classifier.PIIModel.PIIMappingPath)
		if err != nil {
			return nil, fmt.Errorf("failed to load PII mapping: %w", err)
		}
		log.Printf("Loaded PII mapping with %d PII types", piiMapping.GetPIITypeCount())
	}

	// Load jailbreak mapping if prompt guard is enabled
	var jailbreakMapping *classification.JailbreakMapping
	if cfg.IsPromptGuardEnabled() {
		jailbreakMapping, err = classification.LoadJailbreakMapping(cfg.PromptGuard.JailbreakMappingPath)
		if err != nil {
			return nil, fmt.Errorf("failed to load jailbreak mapping: %w", err)
		}
		log.Printf("Loaded jailbreak mapping with %d jailbreak types", jailbreakMapping.GetJailbreakTypeCount())
	}

	if !initialized {
		if err := initializeModels(cfg, categoryMapping, piiMapping, jailbreakMapping); err != nil {
			return nil, err
		}
		initialized = true
	}

	categoryDescriptions := cfg.GetCategoryDescriptions()
	log.Printf("Category descriptions: %v", categoryDescriptions)

	// Create semantic cache with config options
	cacheConfig := cache.CacheConfig{
		BackendType:         cache.CacheBackendType(cfg.SemanticCache.BackendType),
		Enabled:             cfg.SemanticCache.Enabled,
		SimilarityThreshold: cfg.GetCacheSimilarityThreshold(),
		MaxEntries:          cfg.SemanticCache.MaxEntries,
		TTLSeconds:          cfg.SemanticCache.TTLSeconds,
		BackendConfigPath:   cfg.SemanticCache.BackendConfigPath,
	}

	// Use default backend type if not specified
	if cacheConfig.BackendType == "" {
		cacheConfig.BackendType = cache.InMemoryCacheType
	}

	semanticCache, err := cache.NewCacheBackend(cacheConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create semantic cache: %w", err)
	}

	if semanticCache.IsEnabled() {
		log.Printf("Semantic cache enabled (backend: %s) with threshold: %.4f, TTL: %d seconds",
			cacheConfig.BackendType, cacheConfig.SimilarityThreshold, cacheConfig.TTLSeconds)
		if cacheConfig.BackendType == cache.InMemoryCacheType {
			log.Printf("In-memory cache max entries: %d", cacheConfig.MaxEntries)
		}
	} else {
		log.Println("Semantic cache is disabled")
	}

	// Create tools database with config options
	toolsThreshold := cfg.BertModel.Threshold // Default to BERT threshold
	if cfg.Tools.SimilarityThreshold != nil {
		toolsThreshold = *cfg.Tools.SimilarityThreshold
	}
	toolsOptions := tools.ToolsDatabaseOptions{
		SimilarityThreshold: toolsThreshold,
		Enabled:             cfg.Tools.Enabled,
	}
	toolsDatabase := tools.NewToolsDatabase(toolsOptions)

	// Load tools from file if enabled and path is provided
	if toolsDatabase.IsEnabled() && cfg.Tools.ToolsDBPath != "" {
		if err := toolsDatabase.LoadToolsFromFile(cfg.Tools.ToolsDBPath); err != nil {
			log.Printf("Warning: Failed to load tools from file %s: %v", cfg.Tools.ToolsDBPath, err)
		}
		log.Printf("Tools database enabled with threshold: %.4f, top-k: %d",
			toolsThreshold, cfg.Tools.TopK)
	} else {
		log.Println("Tools database is disabled")
	}

	// Create utility components
	piiChecker := pii.NewPolicyChecker(cfg, cfg.ModelConfig)
	classifier := classification.NewClassifier(cfg, categoryMapping, piiMapping, jailbreakMapping)

	// Create global classification service for API access
	services.NewClassificationService(classifier, cfg)

	// Initialize jailbreak classifier if enabled
	if jailbreakMapping != nil {
		err = classifier.InitializeJailbreakClassifier()
		if err != nil {
			return nil, fmt.Errorf("failed to initialize jailbreak classifier: %w", err)
		}
	}

	router := &OpenAIRouter{
		Config:               cfg,
		CategoryDescriptions: categoryDescriptions,
		Classifier:           classifier,
		PIIChecker:           piiChecker,
		Cache:                semanticCache,
		ToolsDatabase:        toolsDatabase,
		pendingRequests:      make(map[string][]byte),
	}

	// Log reasoning configuration after router is created
	router.logReasoningConfiguration()

	return router, nil
}

// initializeModels initializes the BERT and classifier models
func initializeModels(cfg *config.RouterConfig, categoryMapping *classification.CategoryMapping, piiMapping *classification.PIIMapping, jailbreakMapping *classification.JailbreakMapping) error {
	// Initialize the BERT model for similarity search
	err := candle_binding.InitModel(cfg.BertModel.ModelID, cfg.BertModel.UseCPU)
	if err != nil {
		return fmt.Errorf("failed to initialize BERT model: %w", err)
	}

	// Initialize the classifier model if enabled
	if categoryMapping != nil {
		// Get the number of categories from the mapping
		numClasses := categoryMapping.GetCategoryCount()
		if numClasses < 2 {
			log.Printf("Warning: Not enough categories for classification, need at least 2, got %d", numClasses)
		} else {
			// Use the category classifier model
			classifierModelID := cfg.Classifier.CategoryModel.ModelID
			if classifierModelID == "" {
				classifierModelID = cfg.BertModel.ModelID
			}

			if cfg.Classifier.CategoryModel.UseModernBERT {
				// Initialize ModernBERT classifier
				err = candle_binding.InitModernBertClassifier(classifierModelID, cfg.Classifier.CategoryModel.UseCPU)
				if err != nil {
					return fmt.Errorf("failed to initialize ModernBERT classifier model: %w", err)
				}
				log.Printf("Initialized ModernBERT category classifier (classes auto-detected from model)")
			} else {
				// Initialize linear classifier
				err = candle_binding.InitClassifier(classifierModelID, numClasses, cfg.Classifier.CategoryModel.UseCPU)
				if err != nil {
					return fmt.Errorf("failed to initialize classifier model: %w", err)
				}
				log.Printf("Initialized linear category classifier with %d categories", numClasses)
			}
		}
	}

	// Initialize PII classifier if enabled
	if piiMapping != nil {
		// Get the number of PII types from the mapping
		numPIIClasses := piiMapping.GetPIITypeCount()
		if numPIIClasses < 2 {
			log.Printf("Warning: Not enough PII types for classification, need at least 2, got %d", numPIIClasses)
		} else {
			// Use the PII classifier model
			piiClassifierModelID := cfg.Classifier.PIIModel.ModelID
			if piiClassifierModelID == "" {
				piiClassifierModelID = cfg.BertModel.ModelID
			}

			// Initialize ModernBERT PII token classifier for entity detection
			err = candle_binding.InitModernBertPIITokenClassifier(piiClassifierModelID, cfg.Classifier.PIIModel.UseCPU)
			if err != nil {
				return fmt.Errorf("failed to initialize ModernBERT PII token classifier model: %w", err)
			}
			log.Printf("Initialized ModernBERT PII token classifier for entity detection")
		}
	}

	// Initialize jailbreak classifier if enabled
	if jailbreakMapping != nil {
		// Get the number of jailbreak types from the mapping
		numJailbreakClasses := jailbreakMapping.GetJailbreakTypeCount()
		if numJailbreakClasses < 2 {
			log.Printf("Warning: Not enough jailbreak types for classification, need at least 2, got %d", numJailbreakClasses)
		} else {
			// Use the jailbreak classifier model
			jailbreakClassifierModelID := cfg.PromptGuard.ModelID
			if jailbreakClassifierModelID == "" {
				jailbreakClassifierModelID = cfg.BertModel.ModelID
			}

			if cfg.PromptGuard.UseModernBERT {
				// Initialize ModernBERT jailbreak classifier
				err = candle_binding.InitModernBertJailbreakClassifier(jailbreakClassifierModelID, cfg.PromptGuard.UseCPU)
				if err != nil {
					return fmt.Errorf("failed to initialize ModernBERT jailbreak classifier model: %w", err)
				}
				log.Printf("Initialized ModernBERT jailbreak classifier (classes auto-detected from model)")
			} else {
				// Initialize linear jailbreak classifier
				err = candle_binding.InitJailbreakClassifier(jailbreakClassifierModelID, numJailbreakClasses, cfg.PromptGuard.UseCPU)
				if err != nil {
					return fmt.Errorf("failed to initialize jailbreak classifier model: %w", err)
				}
				log.Printf("Initialized linear jailbreak classifier with %d jailbreak types", numJailbreakClasses)
			}
		}
	}

	return nil
}
