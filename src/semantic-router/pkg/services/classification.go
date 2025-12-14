package services

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/classification"
)

// Global classification service instance
var globalClassificationService *ClassificationService

// ClassificationService provides classification functionality
type ClassificationService struct {
	classifier        *classification.Classifier
	unifiedClassifier *classification.UnifiedClassifier // New unified classifier
	config            *config.RouterConfig
	configMutex       sync.RWMutex // Protects config access
}

// NewClassificationService creates a new classification service
func NewClassificationService(classifier *classification.Classifier, config *config.RouterConfig) *ClassificationService {
	service := &ClassificationService{
		classifier:        classifier,
		unifiedClassifier: nil, // Will be initialized separately
		config:            config,
	}
	// Set as global service for API access
	globalClassificationService = service
	return service
}

// NewUnifiedClassificationService creates a new service with unified classifier
func NewUnifiedClassificationService(unifiedClassifier *classification.UnifiedClassifier, legacyClassifier *classification.Classifier, config *config.RouterConfig) *ClassificationService {
	service := &ClassificationService{
		classifier:        legacyClassifier,
		unifiedClassifier: unifiedClassifier,
		config:            config,
	}
	// Set as global service for API access
	globalClassificationService = service
	return service
}

// NewClassificationServiceWithAutoDiscovery creates a service with auto-discovery
func NewClassificationServiceWithAutoDiscovery(config *config.RouterConfig) (*ClassificationService, error) {
	// Debug: Check current working directory
	wd, _ := os.Getwd()
	observability.Debugf("Debug: Current working directory: %s", wd)
	observability.Debugf("Debug: Attempting to discover models in: ./models")

	// Always try to auto-discover and initialize unified classifier for batch processing
	// Use model path from config, fallback to "./models" if not specified
	modelsPath := "./models"
	if config != nil && config.Classifier.CategoryModel.ModelID != "" {
		// Extract the models directory from the model path
		// e.g., "models/category_classifier_modernbert-base_model" -> "models"
		if idx := strings.Index(config.Classifier.CategoryModel.ModelID, "/"); idx > 0 {
			modelsPath = config.Classifier.CategoryModel.ModelID[:idx]
		}
	}
	unifiedClassifier, ucErr := classification.AutoInitializeUnifiedClassifier(modelsPath)
	if ucErr != nil {
		observability.Infof("Unified classifier auto-discovery failed: %v", ucErr)
	}
	// create legacy classifier
	legacyClassifier, lcErr := createLegacyClassifier(config)
	if lcErr != nil {
		observability.Warnf("Legacy classifier initialization failed: %v", lcErr)
	}
	if unifiedClassifier == nil && legacyClassifier == nil {
		observability.Warnf("No classifier initialized. Using placeholder service.")
	}
	return NewUnifiedClassificationService(unifiedClassifier, legacyClassifier, config), nil
}

// createLegacyClassifier creates a legacy classifier with proper model loading
func createLegacyClassifier(config *config.RouterConfig) (*classification.Classifier, error) {
	// Load category mapping
	var categoryMapping *classification.CategoryMapping

	// Check if we should load categories from MCP server
	// Note: tool_name is optional and will be auto-discovered if not specified
	useMCPCategories := config.Classifier.CategoryModel.ModelID == "" &&
		config.Classifier.MCPCategoryModel.Enabled

	if useMCPCategories {
		// Categories will be loaded from MCP server during initialization
		observability.Infof("Category mapping will be loaded from MCP server")
		// Create empty mapping initially - will be populated during initialization
		categoryMapping = nil
	} else if config.Classifier.CategoryModel.CategoryMappingPath != "" {
		// Load from file as usual
		var err error
		categoryMapping, err = classification.LoadCategoryMapping(config.Classifier.CategoryModel.CategoryMappingPath)
		if err != nil {
			return nil, fmt.Errorf("failed to load category mapping: %w", err)
		}
	}

	// Load PII mapping
	var piiMapping *classification.PIIMapping
	if config.Classifier.PIIModel.PIIMappingPath != "" {
		var err error
		piiMapping, err = classification.LoadPIIMapping(config.Classifier.PIIModel.PIIMappingPath)
		if err != nil {
			return nil, fmt.Errorf("failed to load PII mapping: %w", err)
		}
	}

	// Load jailbreak mapping
	var jailbreakMapping *classification.JailbreakMapping
	if config.PromptGuard.JailbreakMappingPath != "" {
		var err error
		jailbreakMapping, err = classification.LoadJailbreakMapping(config.PromptGuard.JailbreakMappingPath)
		if err != nil {
			return nil, fmt.Errorf("failed to load jailbreak mapping: %w", err)
		}
	}

	// Create classifier
	classifier, err := classification.NewClassifier(config, categoryMapping, piiMapping, jailbreakMapping)
	if err != nil {
		return nil, fmt.Errorf("failed to create classifier: %w", err)
	}

	return classifier, nil
}

// GetGlobalClassificationService returns the global classification service instance
func GetGlobalClassificationService() *ClassificationService {
	return globalClassificationService
}

// HasClassifier returns true if the service has a real classifier (not placeholder)
func (s *ClassificationService) HasClassifier() bool {
	return s.classifier != nil
}

// NewPlaceholderClassificationService creates a placeholder service for API-only mode
func NewPlaceholderClassificationService() *ClassificationService {
	return &ClassificationService{
		classifier: nil, // No classifier - will return placeholder responses
		config:     nil,
	}
}

// IntentRequest represents a request for intent classification
type IntentRequest struct {
	Text    string         `json:"text"`
	Options *IntentOptions `json:"options,omitempty"`
}

// IntentOptions contains options for intent classification
type IntentOptions struct {
	ReturnProbabilities bool    `json:"return_probabilities,omitempty"`
	ConfidenceThreshold float64 `json:"confidence_threshold,omitempty"`
	IncludeExplanation  bool    `json:"include_explanation,omitempty"`
}

// IntentResponse represents the response from intent classification
type IntentResponse struct {
	Classification   Classification     `json:"classification"`
	Probabilities    map[string]float64 `json:"probabilities,omitempty"`
	RecommendedModel string             `json:"recommended_model,omitempty"`
	RoutingDecision  string             `json:"routing_decision,omitempty"`
}

// Classification represents basic classification result
type Classification struct {
	Category         string  `json:"category"`
	Confidence       float64 `json:"confidence"`
	ProcessingTimeMs int64   `json:"processing_time_ms"`
}

// MultimodalClassificationRequest represents a request for multimodal classification
type MultimodalClassificationRequest struct {
	Text        string                 `json:"text"`
	Images      []MultimodalImage      `json:"images,omitempty"`
	ContentType string                 `json:"content_type"` // "text", "image", "multimodal"
	Options     *ClassificationOptions `json:"options,omitempty"`
}

// MultimodalImage represents an image in a multimodal classification request
type MultimodalImage struct {
	Data        string `json:"data"`                  // Base64 encoded image
	URL         string `json:"url,omitempty"`         // Alternative: image URL
	MimeType    string `json:"mime_type"`             // image/jpeg, image/png, etc.
	Description string `json:"description,omitempty"` // Optional text description
}

// ClassificationOptions contains options for multimodal classification
type ClassificationOptions struct {
	ReturnProbabilities bool    `json:"return_probabilities,omitempty"`
	ConfidenceThreshold float64 `json:"confidence_threshold,omitempty"`
	IncludeExplanation  bool    `json:"include_explanation,omitempty"`
}

// ClassifyIntent performs intent classification
func (s *ClassificationService) ClassifyIntent(req IntentRequest) (*IntentResponse, error) {
	start := time.Now()

	if req.Text == "" {
		return nil, fmt.Errorf("text cannot be empty")
	}

	// Check if classifier is available
	if s.classifier == nil {
		// Return placeholder response
		processingTime := time.Since(start).Milliseconds()
		return &IntentResponse{
			Classification: Classification{
				Category:         "general",
				Confidence:       0.5,
				ProcessingTimeMs: processingTime,
			},
			RecommendedModel: "general-model",
			RoutingDecision:  "placeholder_response",
		}, nil
	}

	// Perform classification using the existing classifier
	category, confidence, err := s.classifier.ClassifyCategory(req.Text)
	if err != nil {
		return nil, fmt.Errorf("classification failed: %w", err)
	}

	processingTime := time.Since(start).Milliseconds()

	// Build response
	response := &IntentResponse{
		Classification: Classification{
			Category:         category,
			Confidence:       confidence,
			ProcessingTimeMs: processingTime,
		},
	}

	// Add probabilities if requested
	if req.Options != nil && req.Options.ReturnProbabilities {
		// TODO: Implement probability extraction from classifier
		response.Probabilities = map[string]float64{
			category: confidence,
		}
	}

	// Add recommended model based on category
	if model := s.getRecommendedModel(category, confidence); model != "" {
		response.RecommendedModel = model
	}

	// Determine routing decision
	response.RoutingDecision = s.getRoutingDecision(confidence, req.Options)

	return response, nil
}

// ClassifyMultimodal performs intent classification on multimodal input (text + images).
//
// PRODUCTION APPROACH: Uses vision transformer embeddings for native image processing.
// Flow:
//  1. Extract text embedding from text input
//  2. Extract image embeddings from images using vision transformer (CLIP)
//  3. Fuse text + image embeddings
//  4. Classify using fused embedding via similarity matching
//
// Falls back to prototype approach (Ollama text descriptions) if vision transformer is unavailable.
func (s *ClassificationService) ClassifyMultimodal(req MultimodalClassificationRequest) (*IntentResponse, error) {
	start := time.Now()

	// Validate input
	if req.Text == "" && len(req.Images) == 0 {
		return nil, fmt.Errorf("either text or images must be provided")
	}

	// Try embedding-based approach first
	if len(req.Images) > 0 {
		// 1. Get text embedding
		var textEmb []float32
		var err error

		if req.Text != "" {
			// Get text embedding - try GetEmbeddingWithDim first (requires ModelFactory)
			// If ModelFactory not initialized, fall back to GetEmbeddingDefault (BERT, 384-dim)
			// We'll project to 512 dimensions in fuseTextAndImageEmbeddings
			textEmb, err = candle_binding.GetEmbeddingWithDim(req.Text, 0.5, 0.5, 512)
			if err != nil {
				observability.Debugf("GetEmbeddingWithDim failed (ModelFactory may not be initialized): %v. Using GetEmbeddingDefault (BERT, 384-dim)...", err)
				// Fallback to BERT embedding (384 dimensions)
				// Will be projected to 512 in fusion step
				textEmb, err = candle_binding.GetEmbeddingDefault(req.Text)
				if err != nil {
					observability.Warnf("Failed to get text embedding: %v. Falling back to prototype approach.", err)
					return s.classifyMultimodalWithOllama(req)
				}
			}
			observability.Debugf("Extracted text embedding (dim: %d)", len(textEmb))
		} else {
			// If no text, use zero embedding (will be dominated by image)
			// Create zero embedding with 512 dimensions to match CLIP
			textEmb = make([]float32, 512)
		}

		// 2. Get image embeddings
		observability.Infof("Processing %d image(s) with vision transformer", len(req.Images))
		imageEmbs, err := s.extractImageEmbeddings(req.Images)
		if err != nil {
			observability.Warnf("Failed to extract image embeddings: %v. Falling back to prototype approach.", err)
			return s.classifyMultimodalWithOllama(req)
		}
		observability.Infof("Successfully extracted %d image embedding(s)", len(imageEmbs))

		// 3. Fuse embeddings
		fusedEmb := s.fuseTextAndImageEmbeddings(textEmb, imageEmbs)
		observability.Debugf("Fused embedding dimension: %d", len(fusedEmb))

		// 4. Classify using embedding
		category, confidence, err := s.classifyWithEmbedding(fusedEmb)
		if err != nil {
			observability.Warnf("Embedding-based classification failed: %v. Falling back to prototype approach.", err)
			return s.classifyMultimodalWithOllama(req)
		}

		processingTime := time.Since(start).Milliseconds()
		processingTimeSeconds := float64(processingTime) / 1000.0

		observability.Infof("Multimodal classification result: category=%s, confidence=%.3f, time=%dms",
			category, confidence, processingTime)

		// Record metrics for quantitative analysis
		metrics.RecordCategoryClassification(category)
		metrics.RecordClassificationConfidence(category, "multimodal", confidence)
		metrics.RecordClassifierLatency("multimodal", processingTimeSeconds)

		// 5. Build response
		response := &IntentResponse{
			Classification: Classification{
				Category:         category,
				Confidence:       confidence,
				ProcessingTimeMs: processingTime,
			},
		}

		// Add recommended model
		if model := s.getRecommendedModel(category, confidence); model != "" {
			response.RecommendedModel = model
		}

		// Add routing decision
		response.RoutingDecision = s.getRoutingDecision(confidence, convertClassificationOptions(req.Options))

		return response, nil
	}

	// No images, just text - use standard classification
	if req.Text != "" {
		return s.ClassifyIntent(IntentRequest{
			Text:    req.Text,
			Options: convertClassificationOptions(req.Options),
		})
	}

	return nil, fmt.Errorf("no text or images provided")
}

// classifyMultimodalWithOllama is the fallback approach using Ollama
func (s *ClassificationService) classifyMultimodalWithOllama(req MultimodalClassificationRequest) (*IntentResponse, error) {
	text := req.Text

	var imageDescriptions []string
	if len(req.Images) > 0 {
		observability.Infof("Processing %d image(s) with Ollama (fallback)", len(req.Images))
		descriptions, err := s.extractImageFeatures(req.Images)
		if err != nil {
			observability.Errorf("Failed: ", err)
		} else if len(descriptions) == 0 {
			observability.Warnf("no error occurred")
		} else {
			imageDescriptions = descriptions
			observability.Infof("Successfully extracted %d", len(descriptions))
		}
	}

	fusedText := s.fuseTextAndImageFeatures(text, imageDescriptions)
	observability.Debugf("Fused text length: %d chars", len(fusedText))

	intentReq := IntentRequest{
		Text:    fusedText,
		Options: convertClassificationOptions(req.Options),
	}

	intentResp, err := s.ClassifyIntent(intentReq)
	if err != nil {
		return nil, fmt.Errorf("classification failed: %w", err)
	}

	observability.Infof("Classification result: category=%s, confidence=%.3f",
		intentResp.Classification.Category, intentResp.Classification.Confidence)

	return intentResp, nil
}

// extractImageFeatures extracts text descriptions from images using Ollama (fallback approach).
// Use extractImageEmbeddings() for the production embedding-based approach.
func (s *ClassificationService) extractImageFeatures(images []MultimodalImage) ([]string, error) {
	var descriptions []string
	ollamaURL := "http://127.0.0.1:11434/api/generate" // Hardcoded for prototype
	ollamaModel := "llava:7b"

	for i, img := range images {
		if img.Data == "" {
			observability.Warnf("Skipping image %d with empty data", i+1)
			continue
		}

		cleanData := strings.ReplaceAll(strings.TrimSpace(img.Data), "\n", "")
		observability.Debugf("Sending image %d to Ollama (data length: %d chars, mime_type: %s)", i+1, len(cleanData), img.MimeType)

		body := map[string]interface{}{
			"model":  ollamaModel,
			"prompt": "Describe the image briefly for context.",
			"images": []string{cleanData},
		}

		payload, err := json.Marshal(body)
		if err != nil {
			observability.Errorf("Failed to marshal Ollama request: %v", err)
			return nil, fmt.Errorf("failed to prepare image request: %w", err)
		}

		observability.Debugf("Calling Ollama at %s with model %s", ollamaURL, ollamaModel)
		resp, err := http.Post(ollamaURL, "application/json", bytes.NewReader(payload))
		if err != nil {
			observability.Errorf("Ollama request failed: %v", err)
			return nil, fmt.Errorf("failed to connect to Ollama at %s: %w", ollamaURL, err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			bodyBytes, _ := io.ReadAll(resp.Body)
			observability.Errorf("Ollama returned status %d: %s", resp.StatusCode, string(bodyBytes))
			return nil, fmt.Errorf("ollama returned status %d: %s", resp.StatusCode, string(bodyBytes))
		}

		var fullResponse string
		decoder := json.NewDecoder(resp.Body)
		for decoder.More() {
			var chunk map[string]interface{}
			if err := decoder.Decode(&chunk); err != nil {
				observability.Warnf("Error decoding Ollama chunk: %v", err)
				break
			}
			if part, ok := chunk["response"].(string); ok {
				fullResponse += part
			}
		}

		if fullResponse != "" {
			descriptions = append(descriptions, fullResponse)
			observability.Infof("Image processed successfully, description length: %d chars", len(fullResponse))
		} else {
			observability.Warnf("Ollama returned empty response for image")
		}
	}

	return descriptions, nil
}

func (s *ClassificationService) fuseTextAndImageFeatures(text string, imageDescriptions []string) string {
	if len(imageDescriptions) == 0 {
		return text
	}

	var fused strings.Builder
	if text != "" {
		fused.WriteString(text)
	}

	for i, desc := range imageDescriptions {
		if desc != "" {
			if fused.Len() > 0 {
				fused.WriteString("\n")
			}
			if len(imageDescriptions) > 1 {
				fused.WriteString(fmt.Sprintf("Visual context %d: %s", i+1, desc))
			} else {
				fused.WriteString("Visual context: " + desc)
			}
		}
	}

	return fused.String()
}

// extractImageEmbeddings extracts 512-dimensional embedding vectors from images using CLIP Vision Transformer.
// Each image is preprocessed (decoded, resized to 224x224, normalized) and passed through CLIP ViT.
// Returns one embedding per successfully processed image.
func (s *ClassificationService) extractImageEmbeddings(images []MultimodalImage) ([][]float32, error) {
	var embeddings [][]float32

	for i, img := range images {
		if img.Data == "" {
			observability.Warnf("Skipping image %d with empty data", i+1)
			continue
		}

		// Decode base64 image
		imageBytes, err := base64.StdEncoding.DecodeString(img.Data)
		if err != nil {
			observability.Warnf("Failed to decode base64 image %d: %v", i+1, err)
			continue // Skip this image, continue with others
		}

		// Get image embedding from Candle vision transformer
		embedding, err := candle_binding.GetImageEmbedding(imageBytes, img.MimeType)
		if err != nil {
			observability.Warnf("Failed to extract embedding for image %d: %v", i+1, err)
			continue // Skip this image, continue with others
		}

		embeddings = append(embeddings, embedding)
		observability.Debugf("Successfully extracted embedding for image %d (dim: %d)", i+1, len(embedding))
	}

	if len(embeddings) == 0 {
		return nil, fmt.Errorf("failed to extract embeddings from any image")
	}

	return embeddings, nil
}

// fuseTextAndImageEmbeddings combines text and image embeddings into a unified 512-dimensional embedding.
//
// Process:
//  1. Average multiple image embeddings element-wise (if multiple images)
//  2. Project both text and image embeddings to 512 dimensions (pad or truncate)
//  3. Weighted combination: fused = 0.5 * textEmb + 0.5 * imageEmb
//  4. L2 normalization to unit length for cosine similarity
//
// Returns normalized fused embedding ready for similarity matching.
func (s *ClassificationService) fuseTextAndImageEmbeddings(
	textEmbedding []float32,
	imageEmbeddings [][]float32,
) []float32 {
	if len(imageEmbeddings) == 0 {
		return textEmbedding
	}

	imageAvg := s.averageEmbeddings(imageEmbeddings)

	// Align dimensions to 512 (CLIP's native dimension) for consistent fusion
	targetDim := 512
	if len(textEmbedding) != len(imageAvg) {
		observability.Warnf("Dimension mismatch: text=%d, image=%d. Projecting both to %d dimensions.",
			len(textEmbedding), len(imageAvg), targetDim)

		if len(textEmbedding) != targetDim {
			if len(textEmbedding) < targetDim {
				padded := make([]float32, targetDim)
				copy(padded, textEmbedding)
				textEmbedding = padded
			} else {
				textEmbedding = textEmbedding[:targetDim]
			}
		}

		if len(imageAvg) != targetDim {
			if len(imageAvg) < targetDim {
				padded := make([]float32, targetDim)
				copy(padded, imageAvg)
				imageAvg = padded
			} else {
				imageAvg = imageAvg[:targetDim]
			}
		}
	}

	// Weighted combination: 50% text, 50% image (balanced multimodal fusion)
	textWeight := float32(0.5)
	imageWeight := float32(0.5)

	fused := make([]float32, len(textEmbedding))
	for i := range fused {
		fused[i] = textWeight*textEmbedding[i] + imageWeight*imageAvg[i]
	}

	// Normalize the fused embedding
	norm := float32(0.0)
	for _, v := range fused {
		norm += v * v
	}
	norm = float32(math.Sqrt(float64(norm)))
	if norm > 0 {
		for i := range fused {
			fused[i] /= norm
		}
	}

	return fused
}

// averageEmbeddings computes element-wise average of multiple embeddings.
// Used to combine multiple image embeddings into a single representative embedding.
func (s *ClassificationService) averageEmbeddings(embeddings [][]float32) []float32 {
	if len(embeddings) == 0 {
		return nil
	}

	dim := len(embeddings[0])
	avg := make([]float32, dim)

	for _, emb := range embeddings {
		if len(emb) != dim {
			observability.Warnf("Embedding dimension mismatch: expected %d, got %d", dim, len(emb))
			continue
		}
		for i := range avg {
			avg[i] += emb[i]
		}
	}

	count := float32(len(embeddings))
	for i := range avg {
		avg[i] /= count
	}

	return avg
}

// classifyWithEmbedding classifies a query embedding by computing cosine similarity
// against all category description embeddings and returning the best match.
//
// Process:
//  1. For each category in config, extract embedding from its description text
//  2. Calculate cosine similarity: dot(queryEmb, categoryEmb) (both normalized)
//  3. Return category with highest similarity score and its confidence
func (s *ClassificationService) classifyWithEmbedding(embedding []float32) (string, float64, error) {
	if s.config == nil || len(s.config.Categories) == 0 {
		return "", 0.0, fmt.Errorf("no categories configured")
	}

	bestCategory := ""
	bestScore := -1.0
	secondBestScore := -1.0

	for _, category := range s.config.Categories {
		// Skip if no description
		if category.Description == "" {
			continue
		}

		// Extract embedding for category description (try Qwen3/Gemma, fallback to BERT)
		targetDim := len(embedding)
		catEmb, err := candle_binding.GetEmbeddingWithDim(category.Description, 0.5, 0.5, targetDim)
		if err != nil {
			catEmb, err = candle_binding.GetEmbeddingDefault(category.Description)
			if err != nil {
				observability.Debugf("Failed to get embedding for category %s: %v", category.Name, err)
				continue
			}
		}

		// Project category embedding to match query embedding dimension
		if len(catEmb) != len(embedding) {
			observability.Debugf("Dimension mismatch for category %s: cat=%d, query=%d. Projecting...",
				category.Name, len(catEmb), len(embedding))
			if len(catEmb) < len(embedding) {
				padded := make([]float32, len(embedding))
				copy(padded, catEmb)
				catEmb = padded
			} else {
				catEmb = catEmb[:len(embedding)]
			}
		}

		// Compute cosine similarity (both embeddings are normalized, so this is just dot product)
		similarity := cosineSimilarity(embedding, catEmb)

		// Track similarity score for this category (real embedding quality metric)
		metrics.RecordClassificationConfidence(category.Name, "multimodal_similarity", float64(similarity))

		// Track best and second-best for margin calculation
		if float64(similarity) > bestScore {
			secondBestScore = bestScore
			bestScore = float64(similarity)
			bestCategory = category.Name
		} else if float64(similarity) > secondBestScore {
			secondBestScore = float64(similarity)
		}
	}

	// Track classification margin (gap between best and second-best similarity)
	if bestScore > 0 && secondBestScore >= 0 {
		margin := bestScore - secondBestScore
		observability.Debugf("Classification margin: %.4f (best=%.4f, second=%.4f)", margin, bestScore, secondBestScore)
		// Note: We could add a specific margin metric here if needed
	}

	if bestCategory == "" {
		return "", 0.0, fmt.Errorf("no matching category found")
	}

	return bestCategory, float64(bestScore), nil
}

// cosineSimilarity calculates cosine similarity between two normalized embeddings.
// Since both embeddings are L2-normalized, this is equivalent to dot product.
// Returns value in range [-1, 1], typically [0, 1] for normalized embeddings.
func cosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0.0
	}

	var dotProduct float32
	var normA, normB float32

	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	norm := float32(math.Sqrt(float64(normA * normB)))
	if norm == 0 {
		return 0.0
	}

	return dotProduct / norm
}

func convertClassificationOptions(opts *ClassificationOptions) *IntentOptions {
	if opts == nil {
		return nil
	}
	return &IntentOptions{
		ReturnProbabilities: opts.ReturnProbabilities,
		ConfidenceThreshold: opts.ConfidenceThreshold,
		IncludeExplanation:  opts.IncludeExplanation,
	}
}

// PIIRequest represents a request for PII detection
type PIIRequest struct {
	Text    string      `json:"text"`
	Options *PIIOptions `json:"options,omitempty"`
}

// PIIOptions contains options for PII detection
type PIIOptions struct {
	EntityTypes         []string `json:"entity_types,omitempty"`
	ConfidenceThreshold float64  `json:"confidence_threshold,omitempty"`
	ReturnPositions     bool     `json:"return_positions,omitempty"`
	MaskEntities        bool     `json:"mask_entities,omitempty"`
}

// PIIResponse represents the response from PII detection
type PIIResponse struct {
	HasPII                 bool        `json:"has_pii"`
	Entities               []PIIEntity `json:"entities"`
	MaskedText             string      `json:"masked_text,omitempty"`
	SecurityRecommendation string      `json:"security_recommendation"`
	ProcessingTimeMs       int64       `json:"processing_time_ms"`
}

// PIIEntity represents a detected PII entity
type PIIEntity struct {
	Type        string  `json:"type"`
	Value       string  `json:"value"`
	Confidence  float64 `json:"confidence"`
	StartPos    int     `json:"start_position,omitempty"`
	EndPos      int     `json:"end_position,omitempty"`
	MaskedValue string  `json:"masked_value,omitempty"`
}

// DetectPII performs PII detection
func (s *ClassificationService) DetectPII(req PIIRequest) (*PIIResponse, error) {
	start := time.Now()

	if req.Text == "" {
		return nil, fmt.Errorf("text cannot be empty")
	}

	// Check if classifier is available
	if s.classifier == nil {
		// Return placeholder response
		processingTime := time.Since(start).Milliseconds()
		return &PIIResponse{
			HasPII:                 false,
			Entities:               []PIIEntity{},
			SecurityRecommendation: "allow",
			ProcessingTimeMs:       processingTime,
		}, nil
	}

	// Perform PII detection using the existing classifier
	piiTypes, err := s.classifier.ClassifyPII(req.Text)
	if err != nil {
		return nil, fmt.Errorf("PII detection failed: %w", err)
	}

	processingTime := time.Since(start).Milliseconds()

	// Build response
	response := &PIIResponse{
		HasPII:           len(piiTypes) > 0,
		Entities:         []PIIEntity{},
		ProcessingTimeMs: processingTime,
	}

	// Convert PII types to entities (simplified for now)
	for _, piiType := range piiTypes {
		entity := PIIEntity{
			Type:       piiType,
			Value:      "[DETECTED]", // Placeholder - would need actual entity extraction
			Confidence: 0.9,          // Placeholder - would need actual confidence
		}
		response.Entities = append(response.Entities, entity)
	}

	// Set security recommendation
	if response.HasPII {
		response.SecurityRecommendation = "block"
	} else {
		response.SecurityRecommendation = "allow"
	}

	return response, nil
}

// SecurityRequest represents a request for security detection
type SecurityRequest struct {
	Text    string           `json:"text"`
	Options *SecurityOptions `json:"options,omitempty"`
}

// SecurityOptions contains options for security detection
type SecurityOptions struct {
	DetectionTypes   []string `json:"detection_types,omitempty"`
	Sensitivity      string   `json:"sensitivity,omitempty"`
	IncludeReasoning bool     `json:"include_reasoning,omitempty"`
}

// SecurityResponse represents the response from security detection
type SecurityResponse struct {
	IsJailbreak      bool     `json:"is_jailbreak"`
	RiskScore        float64  `json:"risk_score"`
	DetectionTypes   []string `json:"detection_types"`
	Confidence       float64  `json:"confidence"`
	Recommendation   string   `json:"recommendation"`
	Reasoning        string   `json:"reasoning,omitempty"`
	PatternsDetected []string `json:"patterns_detected"`
	ProcessingTimeMs int64    `json:"processing_time_ms"`
}

// CheckSecurity performs security detection
func (s *ClassificationService) CheckSecurity(req SecurityRequest) (*SecurityResponse, error) {
	start := time.Now()

	if req.Text == "" {
		return nil, fmt.Errorf("text cannot be empty")
	}

	// Check if classifier is available
	if s.classifier == nil {
		// Return placeholder response
		processingTime := time.Since(start).Milliseconds()
		return &SecurityResponse{
			IsJailbreak:      false,
			RiskScore:        0.1,
			DetectionTypes:   []string{},
			Confidence:       0.9,
			Recommendation:   "allow",
			PatternsDetected: []string{},
			ProcessingTimeMs: processingTime,
		}, nil
	}

	// Perform jailbreak detection using the existing classifier
	isJailbreak, jailbreakType, confidence, err := s.classifier.CheckForJailbreak(req.Text)
	if err != nil {
		return nil, fmt.Errorf("security detection failed: %w", err)
	}

	processingTime := time.Since(start).Milliseconds()

	// Build response
	response := &SecurityResponse{
		IsJailbreak:      isJailbreak,
		RiskScore:        float64(confidence),
		Confidence:       float64(confidence),
		ProcessingTimeMs: processingTime,
		DetectionTypes:   []string{},
		PatternsDetected: []string{},
	}

	if isJailbreak {
		response.DetectionTypes = append(response.DetectionTypes, jailbreakType)
		response.PatternsDetected = append(response.PatternsDetected, jailbreakType)
		response.Recommendation = "block"
		if req.Options != nil && req.Options.IncludeReasoning {
			response.Reasoning = fmt.Sprintf("Detected %s pattern with confidence %.3f", jailbreakType, confidence)
		}
	} else {
		response.Recommendation = "allow"
	}

	return response, nil
}

// Helper methods
func (s *ClassificationService) getRecommendedModel(category string, _ float64) string {
	// TODO: Implement model recommendation logic based on category
	return fmt.Sprintf("%s-specialized-model", category)
}

func (s *ClassificationService) getRoutingDecision(confidence float64, options *IntentOptions) string {
	threshold := 0.7 // default threshold
	if options != nil && options.ConfidenceThreshold > 0 {
		threshold = options.ConfidenceThreshold
	}

	if confidence >= threshold {
		return "high_confidence_specialized"
	}
	return "low_confidence_general"
}

// UnifiedBatchResponse represents the response from unified batch classification
type UnifiedBatchResponse struct {
	IntentResults    []classification.IntentResult   `json:"intent_results"`
	PIIResults       []classification.PIIResult      `json:"pii_results"`
	SecurityResults  []classification.SecurityResult `json:"security_results"`
	ProcessingTimeMs int64                           `json:"processing_time_ms"`
	TotalTexts       int                             `json:"total_texts"`
}

// ClassifyBatchUnified performs unified batch classification using the new architecture
func (s *ClassificationService) ClassifyBatchUnified(texts []string) (*UnifiedBatchResponse, error) {
	return s.ClassifyBatchUnifiedWithOptions(texts, nil)
}

// ClassifyBatchUnifiedWithOptions performs unified batch classification with options support
func (s *ClassificationService) ClassifyBatchUnifiedWithOptions(texts []string, _ interface{}) (*UnifiedBatchResponse, error) {
	if len(texts) == 0 {
		return nil, fmt.Errorf("texts cannot be empty")
	}

	// Check if unified classifier is available
	if s.unifiedClassifier == nil {
		return nil, fmt.Errorf("unified classifier not initialized")
	}

	start := time.Now()

	// Direct call to unified classifier - no complex scheduling needed!
	results, err := s.unifiedClassifier.ClassifyBatch(texts)
	if err != nil {
		return nil, fmt.Errorf("unified batch classification failed: %w", err)
	}

	// Build response
	response := &UnifiedBatchResponse{
		IntentResults:    results.IntentResults,
		PIIResults:       results.PIIResults,
		SecurityResults:  results.SecurityResults,
		ProcessingTimeMs: time.Since(start).Milliseconds(),
		TotalTexts:       len(texts),
	}

	return response, nil
}

// ClassifyIntent with unified classifier support (backward compatibility)
func (s *ClassificationService) ClassifyIntentUnified(req IntentRequest) (*IntentResponse, error) {
	if s.unifiedClassifier != nil {
		// Use unified classifier for better performance
		results, err := s.ClassifyBatchUnified([]string{req.Text})
		if err != nil {
			return nil, err
		}

		if len(results.IntentResults) == 0 {
			return nil, fmt.Errorf("no classification results")
		}

		// Convert unified result to legacy format
		intentResult := results.IntentResults[0]

		// Build probabilities map if available
		var probabilities map[string]float64
		if len(intentResult.Probabilities) > 0 && req.Options != nil && req.Options.ReturnProbabilities {
			probabilities = make(map[string]float64)
			// For now, just include the main category probability
			probabilities[intentResult.Category] = float64(intentResult.Confidence)
		}

		return &IntentResponse{
			Classification: Classification{
				Category:         intentResult.Category,
				Confidence:       float64(intentResult.Confidence),
				ProcessingTimeMs: results.ProcessingTimeMs,
			},
			Probabilities:    probabilities,
			RecommendedModel: s.getRecommendedModel(intentResult.Category, float64(intentResult.Confidence)),
			RoutingDecision:  s.getRoutingDecision(float64(intentResult.Confidence), req.Options),
		}, nil
	}

	// Fallback to legacy classifier
	return s.ClassifyIntent(req)
}

// ClassifyPIIUnified performs PII detection using unified classifier
func (s *ClassificationService) ClassifyPIIUnified(texts []string) ([]classification.PIIResult, error) {
	if s.unifiedClassifier == nil {
		return nil, fmt.Errorf("unified classifier not initialized")
	}

	results, err := s.ClassifyBatchUnified(texts)
	if err != nil {
		return nil, err
	}

	return results.PIIResults, nil
}

// ClassifySecurityUnified performs security detection using unified classifier
func (s *ClassificationService) ClassifySecurityUnified(texts []string) ([]classification.SecurityResult, error) {
	if s.unifiedClassifier == nil {
		return nil, fmt.Errorf("unified classifier not initialized")
	}

	results, err := s.ClassifyBatchUnified(texts)
	if err != nil {
		return nil, err
	}

	return results.SecurityResults, nil
}

// HasUnifiedClassifier returns true if the service has a unified classifier
func (s *ClassificationService) HasUnifiedClassifier() bool {
	return s.unifiedClassifier != nil && s.unifiedClassifier.IsInitialized()
}

// GetUnifiedClassifierStats returns statistics about the unified classifier
func (s *ClassificationService) GetUnifiedClassifierStats() map[string]interface{} {
	if s.unifiedClassifier == nil {
		return map[string]interface{}{
			"available": false,
		}
	}

	stats := s.unifiedClassifier.GetStats()
	stats["available"] = true
	return stats
}

// GetConfig returns the current configuration
func (s *ClassificationService) GetConfig() *config.RouterConfig {
	s.configMutex.RLock()
	defer s.configMutex.RUnlock()
	return s.config
}

// UpdateConfig updates the configuration
func (s *ClassificationService) UpdateConfig(newConfig *config.RouterConfig) {
	s.configMutex.Lock()
	defer s.configMutex.Unlock()
	s.config = newConfig
	// Update the global config as well
	config.ReplaceGlobalConfig(newConfig)
}
