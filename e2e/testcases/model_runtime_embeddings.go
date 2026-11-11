package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/modelruntime"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("model-runtime-embeddings", pkgtestcases.TestCase{
		Description: "The embedding deployment a recall store embeds with starts and embeds text",
		Tags:        []string{"model-runtime", "embeddings"},
		Fn:          testModelRuntimeEmbeddings,
	})
}

const mrEmbeddingText = "The tower was completed in 1889."

// testModelRuntimeEmbeddings waits for the embedding deployment, which the
// Router starts only because the vector store recalls with it, and asks its
// worker to embed a text: the fixture's weights are random, so a vector of
// the card's width is the expectation.
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
	dimension, err := checkRuntimeEmbedding(ctx, runtime)
	if err != nil {
		return fmt.Errorf("embedding: %w", err)
	}
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{"embedding_dimension": dimension})
	}
	return nil
}

func checkRuntimeEmbedding(ctx context.Context, runtime *modelruntime.Client) (int, error) {
	direct, err := runtime.Embed(ctx, modelruntime.EmbeddingsRequest{Model: mrEmbeddingDeployment, Input: []string{mrEmbeddingText}})
	if err != nil {
		return 0, err
	}
	if len(direct.Data) != 1 || direct.Data[0].Error != "" || len(direct.Data[0].Embedding) == 0 {
		return 0, fmt.Errorf("the runtime did not embed the text: %+v", direct.Data)
	}
	return len(direct.Data[0].Embedding), nil
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
