package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"strings"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/modelruntime"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("model-runtime-embeddings", pkgtestcases.TestCase{
		Description: "The Router's embedding service returns the vectors of the runtime serving it",
		Tags:        []string{"model-runtime", "embeddings"},
		Fn:          testModelRuntimeEmbeddings,
	})
}

const mrVectorTolerance = 1e-5

const mrEmbeddingText = "The tower was completed in 1889."

// testModelRuntimeEmbeddings asks the Router's embedding API, then the
// deployment's own worker directly: the fixtures' weights are random, so the
// runtime's answer is the expectation.
func testModelRuntimeEmbeddings(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	session, err := openRouterRuntimeSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()
	if err = session.waitReady(ctx, mrEmbeddingDeployment); err != nil {
		return err
	}
	runtime, _, err := session.managed(ctx, mrEmbeddingDeployment)
	if err != nil {
		return err
	}
	dimension, err := checkRouterEmbedding(ctx, session, runtime)
	if err != nil {
		return fmt.Errorf("embedding: %w", err)
	}
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{"embedding_dimension": dimension})
	}
	return nil
}

func checkRouterEmbedding(ctx context.Context, session *modelRuntimeSession, runtime *modelruntime.Client) (int, error) {
	var routed struct {
		Embeddings []struct {
			Embedding []float64 `json:"embedding"`
			Dimension int       `json:"dimension"`
		} `json:"embeddings"`
	}
	if err := session.postAPI(ctx, "/api/v1/diagnostics/embeddings", map[string]interface{}{"texts": []string{mrEmbeddingText}}, &routed); err != nil {
		return 0, err
	}
	if len(routed.Embeddings) != 1 {
		return 0, fmt.Errorf("the Router returned %d embeddings for one text", len(routed.Embeddings))
	}
	got := routed.Embeddings[0]
	direct, err := runtime.Embed(ctx, modelruntime.EmbeddingsRequest{Model: mrEmbeddingDeployment, Input: []string{mrEmbeddingText}, Dimensions: got.Dimension})
	if err != nil {
		return 0, err
	}
	if len(direct.Data) != 1 || direct.Data[0].Error != "" {
		return 0, fmt.Errorf("the runtime did not embed the text: %+v", direct.Data)
	}
	if err := sameVector(got.Embedding, direct.Data[0].Embedding); err != nil {
		return 0, err
	}
	return got.Dimension, nil
}

func sameVector(got, want []float64) error {
	if len(got) != len(want) || len(got) == 0 {
		return fmt.Errorf("length %d, the runtime's %d", len(got), len(want))
	}
	for index := range got {
		if math.Abs(got[index]-want[index]) > mrVectorTolerance {
			return fmt.Errorf("element %d is %.6f, the runtime's %.6f", index, got[index], want[index])
		}
	}
	return nil
}

// postAPI sends a JSON body to the Router's management API and decodes the reply.
func (s *modelRuntimeSession) postAPI(ctx context.Context, path string, body, into interface{}) error {
	payload, err := json.Marshal(body)
	if err != nil {
		return err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, s.api.URL(path), bytes.NewReader(payload))
	if err != nil {
		return err
	}
	request.Header.Set("Content-Type", "application/json")
	return s.callAPI(request, path, into)
}

// getAPI reads one resource of the Router's management API.
func (s *modelRuntimeSession) getAPI(ctx context.Context, path string, into interface{}) error {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, s.api.URL(path), nil)
	if err != nil {
		return err
	}
	return s.callAPI(request, path, into)
}

func (s *modelRuntimeSession) callAPI(request *http.Request, path string, into interface{}) error {
	response, err := s.api.HTTPClient(mrRequestTimeout).Do(request)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	data, err := io.ReadAll(response.Body)
	if err != nil {
		return err
	}
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("%s returned %d: %s", path, response.StatusCode, strings.TrimSpace(string(data)))
	}
	return json.Unmarshal(data, into)
}
