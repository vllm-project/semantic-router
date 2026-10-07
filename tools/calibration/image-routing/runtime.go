package main

import (
	"bytes"
	"context"
	"fmt"
	"image/png"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

const omniDeployment = "omni-calibration"

// runtimeImageBudget is the largest image the managed runtime's request body
// accepts once base64-encoded, leaving room for the JSON envelope.
const runtimeImageBudget = modelservice.ManagedRequestBytes*3/4 - 64<<10

// omniEmbedding serves the Omni snapshot from a model runtime this process
// starts (VLLM_SRUN_COMMAND, else vllm-srun on PATH) and returns
// the production embedding provider over it. stop closes the provider and the
// runtime.
func omniEmbedding(ctx context.Context, snapshot string, maxTokens int) (provider omniProvider, stop func(), err error) {
	deployment := config.ModelDeployment{
		Provider: config.ModelRuntimeProvider,
		Device:   "cpu",
		Artifact: snapshot,
		Input:    config.ModelInputBudget{MaxTokens: maxTokens, Overflow: "reject"},
	}.WithDefaults()
	manager := modelservice.NewManager()
	lease, err := manager.AcquireDeployments(map[string]config.ModelDeployment{omniDeployment: deployment})
	if err != nil {
		_ = manager.Shutdown(ctx)
		return omniProvider{}, nil, err
	}
	shutdown := func() {
		_ = lease.Close()
		_ = manager.Shutdown(context.Background())
	}
	embedding, err := serving.New(lease, nil).Embedding(ctx, config.ResolvedModelBinding{
		Recipe:     "image-calibration",
		Name:       "embedding",
		Binding:    config.ModelBinding{Deployment: omniDeployment, Contract: "embedding.v1"},
		Deployment: deployment,
	}, 0, 0)
	if err != nil {
		shutdown()
		return omniProvider{}, nil, err
	}
	return omniProvider{embedding}, func() {
		_ = embedding.Close()
		shutdown()
	}, nil
}

// omniProvider sends a fixture larger than the request budget as the same
// pixels re-encoded losslessly, so every fixture keeps its calibrated score.
type omniProvider struct{ *serving.EmbeddingProvider }

func (p omniProvider) EmbedImage(ctx context.Context, data []byte, dimension int) ([]float32, error) {
	if len(data) > runtimeImageBudget {
		smaller, err := recompressPNG(data)
		if err != nil {
			return nil, err
		}
		data = smaller
	}
	return p.EmbeddingProvider.EmbedImage(ctx, data, dimension)
}

func recompressPNG(data []byte) ([]byte, error) {
	img, err := png.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("a %d-byte image exceeds the runtime request budget (%d) and is not a PNG to re-encode losslessly: %w", len(data), runtimeImageBudget, err)
	}
	var out bytes.Buffer
	if err := (&png.Encoder{CompressionLevel: png.BestCompression}).Encode(&out, img); err != nil {
		return nil, err
	}
	if out.Len() > runtimeImageBudget {
		return nil, fmt.Errorf("a %d-byte PNG is still %d bytes after lossless re-encoding, over the runtime request budget (%d)", len(data), out.Len(), runtimeImageBudget)
	}
	return out.Bytes(), nil
}
