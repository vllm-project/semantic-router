//go:build !windows && cgo

package apiserver

import (
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

type layerMetadataProvider struct{ embedding.Provider }

func (layerMetadataProvider) Backend() string { return "ort" }
func (layerMetadataProvider) Dimension() int  { return 768 }
func (layerMetadataProvider) EmbeddingInfo() embedding.ModelInfo {
	return embedding.ModelInfo{Artifact: "prepared-artifact", Layers: []int{6, 16}}
}

func TestOwnedEmbeddingRequestUsesPreparedLayers(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.MmBertModelPath = "/unused/legacy-model"
	api := &ClassificationAPIServer{config: cfg}
	set := embedding.NewSet(map[string]embedding.Provider{"mmbert": layerMetadataProvider{}}, "mmbert")
	for _, tc := range []struct {
		layer string
		want  bool
	}{{"16", true}, {"22", false}} {
		request := httptest.NewRequest("POST", "/embeddings", strings.NewReader(`{"texts":["hello"],"model":"mmbert","dimension":256,"target_layer":`+tc.layer+`}`))
		recorder := httptest.NewRecorder()
		_, ok := api.parseEmbeddingRequest(recorder, request, set)
		if ok != tc.want {
			t.Fatalf("layer %s accepted=%v, response=%s", tc.layer, ok, recorder.Body.String())
		}
	}
}
