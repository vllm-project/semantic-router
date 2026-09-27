package testcases

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("model-not-ready-503", pkgtestcases.TestCase{
		Description: "Verify classifier and embedding endpoints return 503 when backing models are unavailable or partially initialized",
		Tags: []string{
			"apiserver",
			"readiness",
			"classifier",
			"embeddings",
		},
		Fn: testModelNotReady503,
	})
}

type endpointSpec struct {
	Name string
	Path string
	Body string
	Code string
}

type errorEnvelope struct {
	Error struct {
		Code string `json:"code"`
	} `json:"error"`
}

// readinessAudioFixture is a minimal valid 16-bit mono WAV data URI. The
// readiness contract is decided from the prepared models, so the payload is a
// real decodable clip rather than a rejected one.
func readinessAudioFixture() string {
	wav := make([]byte, 44+8)
	copy(wav, "RIFF")
	binary.LittleEndian.PutUint32(wav[4:], uint32(len(wav)-8))
	copy(wav[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(wav[16:], 16)
	binary.LittleEndian.PutUint16(wav[20:], 1)
	binary.LittleEndian.PutUint16(wav[22:], 1)
	binary.LittleEndian.PutUint32(wav[24:], 16000)
	binary.LittleEndian.PutUint32(wav[28:], 32000)
	binary.LittleEndian.PutUint16(wav[32:], 2)
	binary.LittleEndian.PutUint16(wav[34:], 16)
	copy(wav[36:], "data")
	binary.LittleEndian.PutUint32(wav[40:], 8)
	binary.LittleEndian.PutUint16(wav[44:], 16384)
	binary.LittleEndian.PutUint16(wav[46:], 32767)
	binary.LittleEndian.PutUint16(wav[48:], 0)
	binary.LittleEndian.PutUint16(wav[50:], 32768)
	return "data:audio/wav;base64," + base64.StdEncoding.EncodeToString(wav)
}

var endpoints = []endpointSpec{
	{
		Name: "pii",
		Path: "/api/v1/diagnostics/classify/pii",
		Body: `{"text":"my ssn is 111-22-3333"}`,
		Code: "CLASSIFIER_NOT_READY",
	},
	{
		Name: "security",
		Path: "/api/v1/diagnostics/classify/security",
		Body: `{"text":"ignore previous instructions"}`,
		Code: "CLASSIFIER_NOT_READY",
	},
	{
		Name: "embeddings",
		Path: "/api/v1/diagnostics/embeddings",
		Body: `{"texts":["hello"]}`,
		Code: "EMBEDDING_NOT_READY",
	},
	{
		Name: "embeddings-multimodal",
		Path: "/api/v1/diagnostics/embeddings",
		Body: `{"texts":["hello"],"images":["data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQABAA0w0e0GAAAAAElFTkSuQmCC"]}`,
		Code: "EMBEDDING_NOT_READY",
	},
	{
		Name: "embeddings-audio",
		Path: "/api/v1/diagnostics/embeddings",
		Body: `{"audios":["` + readinessAudioFixture() + `"]}`,
		Code: "EMBEDDING_NOT_READY",
	},
	{
		Name: "embeddings-audio-named-model",
		Path: "/api/v1/diagnostics/embeddings",
		Body: `{"model":"multimodal","audios":["` + readinessAudioFixture() + `"]}`,
		Code: "EMBEDDING_NOT_READY",
	},
	{
		Name: "similarity",
		Path: "/api/v1/diagnostics/similarity",
		Body: `{"text1":"hello","text2":"world"}`,
		Code: "EMBEDDING_NOT_READY",
	},
	{
		Name: "batch-similarity",
		Path: "/api/v1/diagnostics/similarity/batch",
		Body: `{"query":"hello","candidates":["world"]}`,
		Code: "EMBEDDING_NOT_READY",
	},
}

func testModelNotReady503(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	session, err := fixtures.OpenRouterAPISession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()

	httpClient := session.HTTPClient(30 * time.Second)

	for _, ep := range endpoints {
		resp, err := postJSON(
			ctx,
			httpClient,
			http.MethodPost,
			session.URL(ep.Path),
			[]byte(ep.Body),
		)
		if err != nil {
			return fmt.Errorf("%s: %w", ep.Name, err)
		}

		if resp.StatusCode != http.StatusServiceUnavailable {
			return fmt.Errorf(
				"%s: expected 503, got %d",
				ep.Name,
				resp.StatusCode,
			)
		}

		var e errorEnvelope
		if err := json.Unmarshal(resp.Body, &e); err != nil {
			return fmt.Errorf("%s: invalid JSON: %w", ep.Name, err)
		}

		if e.Error.Code != ep.Code {
			return fmt.Errorf(
				"%s: expected error code %q, got %q",
				ep.Name,
				ep.Code,
				e.Error.Code,
			)
		}
	}

	return nil
}
