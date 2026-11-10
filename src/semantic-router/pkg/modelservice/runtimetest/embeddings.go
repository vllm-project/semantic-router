package runtimetest

import (
	"encoding/binary"
	"math"
	"net/http"
	"slices"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/api"
)

// Embedder makes a model serve /v1/embeddings. A text's vector counts its
// words in hashed buckets (offset by the layer exit), L2-normalized, so equal
// texts embed equally and texts sharing words are similar. Images and audio
// hash their bytes. Dimensions lists the served views, largest first.
type Embedder struct {
	Dimensions []int
	Layers     []int
	Modalities []string
}

// Reranker makes a model serve /v1/rerank. A document's logit is
// 4·(fraction of query words it contains) − 2.
type Reranker struct {
	Default api.RerankExit
	Exits   []api.RerankExit
}

func (r *Runtime) embeddings(body api.EmbeddingsRequest) (int, api.EmbeddingsResponse, *api.ErrorBody) {
	model, status, errBody := r.model(body.Model, "embeddings")
	if status != http.StatusOK {
		return status, api.EmbeddingsResponse{}, errBody
	}
	full := model.Embedding.Dimensions[0]
	dimension, layer := full, 0
	if body.Dimensions != nil {
		dimension = *body.Dimensions
	}
	if body.Layer != nil {
		layer = *body.Layer
	}
	if !slices.Contains(model.Embedding.Dimensions, dimension) || (layer != 0 && !slices.Contains(model.Embedding.Layers, layer)) {
		return http.StatusBadRequest, api.EmbeddingsResponse{}, &api.ErrorBody{Code: "invalid_request", Message: "undeclared dimension or layer"}
	}
	limit, overflow := model.MaxInputTokens, "reject"
	if body.Options != nil {
		if body.Options.MaxTokens != nil && *body.Options.MaxTokens > 0 {
			limit = min(limit, *body.Options.MaxTokens)
		}
		if body.Options.Overflow != nil {
			overflow = string(*body.Options.Overflow)
		}
	}
	parts, ok := embeddingParts(body.Input)
	if !ok {
		return http.StatusBadRequest, api.EmbeddingsResponse{}, &api.ErrorBody{Code: "invalid_request", Message: "unreadable input"}
	}
	response := api.EmbeddingsResponse{Object: "list", Model: model.ID, Data: make([]api.Embedding, len(parts))}
	for i, part := range parts {
		item := api.Embedding{Object: "embedding", Index: i}
		tokens := 2
		var vector []float64
		switch {
		case part.text != nil:
			words := strings.Fields(*part.text)
			tokens += len(words)
			if tokens > limit {
				if overflow == "reject" {
					code := api.ItemError("max_length_exceeded")
					item.Error = &code
					response.Data[i] = item
					continue
				}
				words = words[:max(limit-2, 0)]
			}
			vector = hashedVector(full, layer, words)
		default:
			vector = hashedVector(full, layer, []string{part.data})
		}
		usage := api.InputUsage{Tokens: tokens, ProcessedTokens: min(tokens, limit), Truncated: tokens > limit}
		item.Input = &usage
		response.Usage.PromptTokens += usage.ProcessedTokens
		encoded := encodeVector(normalize(vector[:dimension]), body.EncodingFormat)
		item.Embedding = &encoded
		response.Data[i] = item
	}
	response.Usage.TotalTokens = response.Usage.PromptTokens
	sha := strings.Repeat("e", 64)
	normalized := true
	response.Meta = &api.ResponseMeta{Representation: &api.Representation{ModelSha256: sha, Layer: layer, Dimension: dimension, Normalized: &normalized}}
	return http.StatusOK, response, nil
}

type embeddingPart struct {
	text *string
	data string
}

func embeddingParts(input api.EmbeddingsInput) ([]embeddingPart, bool) {
	if text, err := input.AsInputText(); err == nil {
		return []embeddingPart{{text: &text}}, true
	}
	if texts, err := input.AsTextList(); err == nil {
		parts := make([]embeddingPart, len(texts))
		for i := range texts {
			parts[i] = embeddingPart{text: &texts[i]}
		}
		return parts, true
	}
	list, err := input.AsContentPartList()
	if err != nil {
		return nil, false
	}
	parts := make([]embeddingPart, 0, len(list))
	for _, part := range list {
		switch {
		case part.Type == api.Text:
			text := ""
			if part.Text != nil {
				text = *part.Text
			}
			parts = append(parts, embeddingPart{text: &text})
		case part.Type == api.ImageUrl && part.ImageUrl != nil:
			parts = append(parts, embeddingPart{data: part.ImageUrl.Url})
		case part.Type == api.InputAudio && part.InputAudio != nil:
			parts = append(parts, embeddingPart{data: part.InputAudio.Data})
		default:
			return nil, false
		}
	}
	return parts, true
}

func hashedVector(dimension, layer int, words []string) []float64 {
	vector := make([]float64, dimension)
	for _, word := range words {
		bucket := layer
		for _, octet := range []byte(strings.ToLower(word)) {
			bucket = (bucket*31 + int(octet)) % dimension
		}
		vector[bucket]++
	}
	if len(words) == 0 {
		vector[0] = 1
	}
	return vector
}

func normalize(vector []float64) []float64 {
	var norm float64
	for _, value := range vector {
		norm += value * value
	}
	norm = math.Sqrt(norm)
	out := make([]float64, len(vector))
	for i, value := range vector {
		if norm > 0 {
			out[i] = value / norm
		}
	}
	if norm == 0 {
		out[0] = 1
	}
	return out
}

func encodeVector(vector []float64, format *api.EmbeddingsRequestEncodingFormat) api.EmbeddingVector {
	var encoded api.EmbeddingVector
	if format == nil || *format != "base64" {
		floats := make(api.FloatVector, len(vector))
		for i, value := range vector {
			floats[i] = float32(value)
		}
		_ = encoded.FromFloatVector(floats)
		return encoded
	}
	raw := make([]byte, 4*len(vector))
	for i, value := range vector {
		binary.LittleEndian.PutUint32(raw[4*i:], math.Float32bits(float32(value)))
	}
	_ = encoded.FromBase64Vector(raw)
	return encoded
}

func (r *Runtime) rerank(body api.RerankRequest) (int, api.RerankResponse, *api.ErrorBody) {
	model, status, errBody := r.model(body.Model, "rerank")
	if status != http.StatusOK {
		return status, api.RerankResponse{}, errBody
	}
	exit := model.Rerank.Default
	if body.Layer != nil {
		exit.Layer = *body.Layer
	}
	if body.Dimensions != nil {
		exit.Dimension = *body.Dimensions
	}
	if exit != model.Rerank.Default && !slices.Contains(model.Rerank.Exits, exit) {
		return http.StatusBadRequest, api.RerankResponse{}, &api.ErrorBody{Code: "invalid_request", Message: "undeclared pair-scorer exit"}
	}
	query := strings.Fields(strings.ToLower(body.Query))
	response := api.RerankResponse{Model: model.ID, Results: make([]api.RerankResult, len(body.Documents))}
	for i, document := range body.Documents {
		words := strings.Fields(strings.ToLower(document))
		tokens := len(query) + len(words) + 3
		usage := api.InputUsage{Tokens: tokens, ProcessedTokens: tokens}
		result := api.RerankResult{Index: i, Input: &usage}
		if tokens > model.MaxInputTokens {
			code := api.ItemError("max_length_exceeded")
			result.Error = &code
			response.Results[i] = result
			continue
		}
		shared := 0
		for _, word := range query {
			if slices.Contains(words, word) {
				shared++
			}
		}
		logit := 4*float64(shared)/float64(max(len(query), 1)) - 2
		score := 1 / (1 + math.Exp(-logit))
		result.Logit, result.RelevanceScore = &logit, &score
		response.Usage.InputTokens += tokens
		response.Results[i] = result
	}
	return http.StatusOK, response, nil
}

func embeddingCard(embedder *Embedder) *api.EmbeddingCard {
	modalities := slices.Clone(embedder.Modalities)
	if len(modalities) == 0 {
		modalities = []string{"text"}
	}
	layers := slices.Clone(embedder.Layers)
	if layers == nil {
		layers = []int{}
	}
	return &api.EmbeddingCard{Dimensions: slices.Clone(embedder.Dimensions), Layers: layers, Modalities: modalities, Normalized: true, Pooling: "mean", InputTypes: []string{}}
}

func rerankCard(reranker *Reranker) *api.RerankCard {
	exits := slices.Clone(reranker.Exits)
	if exits == nil {
		exits = []api.RerankExit{}
	}
	return &api.RerankCard{Default: reranker.Default, Exits: exits}
}

// serves reports whether a model answers a surface.
func serves(model Model, surface string) bool {
	switch surface {
	case "classify":
		return len(model.Heads) > 0
	case "embeddings":
		return model.Embedding != nil
	case "rerank":
		return model.Rerank != nil
	}
	return len(model.Heads) == 0 && model.Embedding == nil && model.Rerank == nil
}
