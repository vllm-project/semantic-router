package config

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"strings"
	"sync"
	"time"
)

const (
	defaultHFModelBaseURL = "https://huggingface.co"
	defaultHFModelTTL     = 6 * time.Hour
	maxPlausibleModelCtx  = 1_000_000
)

type cachedRegistryCard struct {
	expiresAt time.Time
	info      ModelRegistryInfo
	ok        bool
}

type modelRegistryCardResolver struct {
	baseURL string
	client  *http.Client
	ttl     time.Duration
	now     func() time.Time

	mu    sync.Mutex
	cache map[string]cachedRegistryCard
}

type hfModelAPIResponse struct {
	ID          string            `json:"id"`
	ModelID     string            `json:"modelId"`
	PipelineTag string            `json:"pipeline_tag"`
	Tags        []string          `json:"tags"`
	CardData    hfModelCardData   `json:"cardData"`
	Safetensors hfSafetensorsInfo `json:"safetensors"`
}

type hfModelCardData struct {
	BaseModel   any      `json:"base_model"`
	License     string   `json:"license"`
	Language    any      `json:"language"`
	Tags        []string `json:"tags"`
	Datasets    any      `json:"datasets"`
	PipelineTag string   `json:"pipeline_tag"`
}

type hfSafetensorsInfo struct {
	Total int64 `json:"total"`
}

type hfModelConfig struct {
	HiddenSize            int               `json:"hidden_size"`
	MaxPositionEmbeddings int               `json:"max_position_embeddings"`
	NumLabels             int               `json:"num_labels"`
	ID2Label              map[string]string `json:"id2label"`
	Label2ID              map[string]int    `json:"label2id"`
}

type hfTokenizerConfig struct {
	ModelMaxLength int `json:"model_max_length"`
}

var defaultRegistryCardResolver = newModelRegistryCardResolver()

func newModelRegistryCardResolver() *modelRegistryCardResolver {
	return &modelRegistryCardResolver{
		baseURL: defaultHFModelBaseURL,
		client:  &http.Client{Timeout: 2 * time.Second},
		ttl:     defaultHFModelTTL,
		now:     time.Now,
		cache:   map[string]cachedRegistryCard{},
	}
}

// GetModelRegistryInfoByPath returns registry metadata for a model path or alias.
// Local registry values are used as the baseline and are overlaid with Hugging Face
// model-card/config metadata when the remote repo is reachable.
func GetModelRegistryInfoByPath(path string) *ModelRegistryInfo {
	spec := GetModelByPath(path)
	if spec == nil {
		return nil
	}

	resolved := defaultRegistryCardResolver.resolve(*spec)
	return &resolved
}

func (r *modelRegistryCardResolver) resolve(spec ModelSpec) ModelRegistryInfo {
	info := spec.RegistryInfo()
	if spec.RepoID == "" {
		return info
	}

	remote, ok := r.lookup(spec.RepoID)
	if !ok {
		return info
	}

	overlayRegistryInfo(&info, remote, spec)
	return info
}

func (r *modelRegistryCardResolver) lookup(repoID string) (ModelRegistryInfo, bool) {
	now := r.now()

	r.mu.Lock()
	if cached, ok := r.cache[repoID]; ok && now.Before(cached.expiresAt) {
		r.mu.Unlock()
		return cached.info, cached.ok
	}
	r.mu.Unlock()

	info, ok := r.fetch(repoID)
	expiresAt := now.Add(r.ttl)

	r.mu.Lock()
	r.cache[repoID] = cachedRegistryCard{
		expiresAt: expiresAt,
		info:      info,
		ok:        ok,
	}
	if ok && info.RepoID != "" && info.RepoID != repoID {
		r.cache[info.RepoID] = cachedRegistryCard{
			expiresAt: expiresAt,
			info:      info,
			ok:        ok,
		}
	}
	r.mu.Unlock()

	return info, ok
}

func (r *modelRegistryCardResolver) fetch(repoID string) (ModelRegistryInfo, bool) {
	apiResp, err := r.fetchJSON(fmt.Sprintf("%s/api/models/%s", strings.TrimSuffix(r.baseURL, "/"), repoID), &hfModelAPIResponse{})
	if err != nil {
		return ModelRegistryInfo{}, false
	}

	api := apiResp.(*hfModelAPIResponse)
	info := ModelRegistryInfo{
		RepoID:       firstNonEmpty(api.ID, api.ModelID, repoID),
		ModelCardURL: fmt.Sprintf("%s/%s", strings.TrimSuffix(r.baseURL, "/"), firstNonEmpty(api.ID, api.ModelID, repoID)),
		PipelineTag:  firstNonEmpty(api.CardData.PipelineTag, api.PipelineTag),
		BaseModel:    firstString(normalizeStringList(api.CardData.BaseModel)...),
		License:      api.CardData.License,
		Languages:    normalizeStringList(api.CardData.Language),
		Datasets:     normalizeStringList(api.CardData.Datasets),
	}

	if len(api.CardData.Tags) > 0 {
		info.Tags = filterRemoteTags(api.CardData.Tags)
	} else {
		info.Tags = filterRemoteTags(api.Tags)
	}
	if api.Safetensors.Total > 0 {
		info.ParameterSize = formatParameterCount(api.Safetensors.Total)
	}

	if readme := r.fetchText(fmt.Sprintf("%s/%s/raw/main/README.md", strings.TrimSuffix(r.baseURL, "/"), repoID)); readme != "" {
		info.Description = extractModelCardDescription(readme)
	}

	var cfg hfModelConfig
	if err := r.fetchJSONInto(fmt.Sprintf("%s/%s/raw/main/config.json", strings.TrimSuffix(r.baseURL, "/"), repoID), &cfg); err == nil {
		info.EmbeddingDim = cfg.HiddenSize
		info.NumClasses = maxInt(cfg.NumLabels, len(cfg.ID2Label), len(cfg.Label2ID))
		if plausibleModelContext(cfg.MaxPositionEmbeddings) {
			info.MaxContextLength = cfg.MaxPositionEmbeddings
		}
	}

	var tokenizer hfTokenizerConfig
	if err := r.fetchJSONInto(fmt.Sprintf("%s/%s/raw/main/tokenizer_config.json", strings.TrimSuffix(r.baseURL, "/"), repoID), &tokenizer); err == nil {
		if plausibleModelContext(tokenizer.ModelMaxLength) {
			info.MaxContextLength = tokenizer.ModelMaxLength
		}
	}

	return info, true
}

func (r *modelRegistryCardResolver) fetchJSON(rawURL string, target any) (any, error) {
	if err := r.fetchJSONInto(rawURL, target); err != nil {
		return nil, err
	}
	return target, nil
}

func (r *modelRegistryCardResolver) fetchJSONInto(rawURL string, target any) error {
	resp, err := r.client.Get(rawURL)
	if err != nil {
		return err
	}
	defer func() {
		_ = resp.Body.Close()
	}()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("unexpected status %d", resp.StatusCode)
	}

	return json.NewDecoder(resp.Body).Decode(target)
}

func (r *modelRegistryCardResolver) fetchText(rawURL string) string {
	resp, err := r.client.Get(rawURL)
	if err != nil {
		return ""
	}
	defer func() {
		_ = resp.Body.Close()
	}()

	if resp.StatusCode != http.StatusOK {
		return ""
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return ""
	}
	return string(body)
}

func overlayRegistryInfo(info *ModelRegistryInfo, remote ModelRegistryInfo, spec ModelSpec) {
	if info == nil {
		return
	}

	info.RepoID = firstNonEmpty(remote.RepoID, info.RepoID)
	info.Description = firstNonEmpty(remote.Description, info.Description)
	if !spec.UsesLoRA {
		info.ParameterSize = firstNonEmpty(remote.ParameterSize, info.ParameterSize)
	}
	if remote.EmbeddingDim > 0 {
		info.EmbeddingDim = remote.EmbeddingDim
	}
	if remote.MaxContextLength > 0 {
		info.MaxContextLength = remote.MaxContextLength
	}
	if remote.NumClasses > 0 {
		info.NumClasses = remote.NumClasses
	}
	info.Tags = mergeOrderedStrings(remote.Tags, info.Tags)
	info.ModelCardURL = firstNonEmpty(remote.ModelCardURL, info.ModelCardURL)
	info.PipelineTag = firstNonEmpty(remote.PipelineTag, info.PipelineTag)
	info.BaseModel = firstNonEmpty(remote.BaseModel, info.BaseModel)
	info.License = firstNonEmpty(remote.License, info.License)
	info.Languages = mergeOrderedStrings(remote.Languages, info.Languages)
	info.Datasets = mergeOrderedStrings(remote.Datasets, info.Datasets)
}

func extractModelCardDescription(readme string) string {
	body := stripModelCardFrontMatter(strings.TrimSpace(readme))
	if body == "" {
		return ""
	}

	return collectModelCardParagraph(strings.Split(body, "\n"))
}

func stripModelCardFrontMatter(body string) string {
	if !strings.HasPrefix(body, "---\n") {
		return body
	}

	if end := strings.Index(body[4:], "\n---\n"); end >= 0 {
		return body[end+8:]
	}

	return body
}

func collectModelCardParagraph(lines []string) string {
	var paragraph []string
	inCodeBlock := false

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if isModelCardCodeFence(trimmed) {
			inCodeBlock = !inCodeBlock
			continue
		}
		if inCodeBlock {
			continue
		}
		if trimmed == "" {
			if len(paragraph) > 0 {
				return strings.Join(paragraph, " ")
			}
			continue
		}
		if shouldSkipModelCardParagraphLine(trimmed) {
			if len(paragraph) > 0 {
				return strings.Join(paragraph, " ")
			}
			continue
		}
		paragraph = append(paragraph, trimmed)
	}

	return strings.Join(paragraph, " ")
}

func isModelCardCodeFence(line string) bool {
	return strings.HasPrefix(line, "```")
}

func shouldSkipModelCardParagraphLine(line string) bool {
	return strings.HasPrefix(line, "#") ||
		strings.HasPrefix(line, "|") ||
		strings.HasPrefix(line, "- ") ||
		strings.HasPrefix(line, "* ")
}

func filterRemoteTags(tags []string) []string {
	var filtered []string
	for _, tag := range tags {
		switch {
		case tag == "":
			continue
		case strings.HasPrefix(tag, "dataset:"):
			continue
		case strings.HasPrefix(tag, "base_model:"):
			continue
		case strings.HasPrefix(tag, "license:"):
			continue
		case strings.HasPrefix(tag, "region:"):
			continue
		default:
			filtered = append(filtered, tag)
		}
	}
	return filtered
}

func mergeOrderedStrings(primary []string, fallback []string) []string {
	if len(primary) == 0 && len(fallback) == 0 {
		return nil
	}

	seen := map[string]struct{}{}
	merged := make([]string, 0, len(primary)+len(fallback))
	for _, value := range append(cloneStrings(primary), fallback...) {
		value = strings.TrimSpace(value)
		if value == "" {
			continue
		}
		if _, ok := seen[value]; ok {
			continue
		}
		seen[value] = struct{}{}
		merged = append(merged, value)
	}
	return merged
}

func normalizeStringList(value any) []string {
	switch typed := value.(type) {
	case string:
		if typed == "" {
			return nil
		}
		return []string{typed}
	case []any:
		values := make([]string, 0, len(typed))
		for _, item := range typed {
			if str, ok := item.(string); ok && str != "" {
				values = append(values, str)
			}
		}
		return values
	case []string:
		return cloneStrings(typed)
	default:
		return nil
	}
}

func formatParameterCount(total int64) string {
	switch {
	case total >= 1_000_000_000:
		whole := total / 1_000_000_000
		tenth := (total % 1_000_000_000) / 100_000_000
		if tenth == 0 {
			return fmt.Sprintf("%dB", whole)
		}
		return fmt.Sprintf("%d.%dB", whole, tenth)
	case total >= 1_000_000:
		return fmt.Sprintf("%dM", total/1_000_000)
	case total >= 1_000:
		return fmt.Sprintf("%dK", total/1_000)
	default:
		return fmt.Sprintf("%d", total)
	}
}

func plausibleModelContext(value int) bool {
	return value > 0 && value <= maxPlausibleModelCtx
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}

func firstString(values ...string) string {
	return firstNonEmpty(values...)
}

func maxInt(values ...int) int {
	maxValue := 0
	for _, value := range values {
		maxValue = int(math.Max(float64(maxValue), float64(value)))
	}
	return maxValue
}
