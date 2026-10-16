package testcases

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"slices"
	"strings"
	"time"

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("vela-omni-contract", pkgtestcases.TestCase{
		Description: "Real Nano and Mini: three modality outputs, input contracts, independent recipes and audio routing",
		Tags:        []string{"real-model", "vela", "multimodal", "embedding", "audio", "recipe-isolation"}, Fn: testVelaOmni,
	})
}

type omniEmbeddingResponse struct {
	Recipe     string `json:"recipe"`
	TotalCount int    `json:"total_count"`
	Embeddings []struct {
		Modality  string    `json:"modality"`
		Model     string    `json:"model_used"`
		Dimension int       `json:"dimension"`
		Vector    []float64 `json:"embedding"`
	} `json:"embeddings"`
}
type omniProbe struct {
	client                           *http.Client
	apiURL, gatewayURL, image, audio string
}

func testVelaOmni(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	apiPort, closeAPI, err := setupRouterAPIConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer closeAPI()
	gatewayPort, closeGateway, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer closeGateway()
	return RunVelaOmniContract(ctx, "http://localhost:"+apiPort, "http://localhost:"+gatewayPort, VelaOmniBudgets{512, 32768}, opts.SetDetails)
}

// VelaOmniBudgets is the authored deployment input budget, not an override of
// the model's published context capacity. GPU deployments specialize their
// execution shape to this budget; the CPU profile uses the full model limits.
type VelaOmniBudgets struct{ Nano, Mini int }

// RunVelaOmniContract runs the same acceptance assertions against a local CLI
// stack or a Kubernetes profile. Endpoint setup does not alter the assertions.
func RunVelaOmniContract(ctx context.Context, apiURL, gatewayURL string, budgets VelaOmniBudgets, setDetails func(map[string]interface{})) error {
	if budgets.Nano < 1 || budgets.Nano > 512 || budgets.Mini < 2048 || budgets.Mini > 32768 {
		return fmt.Errorf("omni acceptance requires Nano budget 1–512 and Mini budget 2048–32768 to exercise >512-token inference")
	}
	image, err := os.ReadFile("e2e/testcases/testdata/image-fixtures/passport_sample.jpg")
	if err != nil {
		return err
	}
	probe := omniProbe{client: &http.Client{Timeout: 120 * time.Second}, apiURL: apiURL, gatewayURL: gatewayURL, image: "data:image/jpeg;base64," + base64.StdEncoding.EncodeToString(image), audio: omniToneWAV()}
	if err := probe.checkEmbeddingOwners(ctx); err != nil {
		return err
	}
	// Revisit Nano after Mini: a process-global last-loaded encoder would return
	// the wrong dimensionality or representation in at least one pass.
	for _, model := range []struct {
		name             string
		dimension, limit int
	}{{"nano", 384, budgets.Nano}, {"mini", 768, budgets.Mini}, {"nano", 384, budgets.Nano}} {
		if err := probe.checkModel(ctx, model.name, model.dimension, model.limit); err != nil {
			return fmt.Errorf("%s: %w", model.name, err)
		}
	}
	if setDetails != nil {
		setDetails(map[string]interface{}{"models": []string{"Vela-1.0-Omni-Nano", "Vela-1.0-Omni-Mini"}, "modalities": []string{"text", "image", "audio"}, "accuracy_rate": 100.0, "minimum_accuracy_rate": 100.0, "recipe_isolation": true, "deployment_input_budgets": budgets})
	}
	return nil
}

func (p omniProbe) checkModel(ctx context.Context, recipe string, dimension, limit int) error {
	if err := p.checkInventory(ctx, recipe, dimension, limit); err != nil {
		return err
	}
	body := map[string]interface{}{"recipe": recipe, "model": "multimodal", "texts": []string{"A high pitched electronic tone."}, "images": []string{p.image}, "audios": []string{p.audio}}
	status, _, data, err := p.post(ctx, p.apiURL+"/api/v1/diagnostics/embeddings", body)
	if err != nil {
		return err
	}
	if status != http.StatusOK {
		return fmt.Errorf("all-modality request status %d: %s", status, data)
	}
	var result omniEmbeddingResponse
	if decodeErr := json.Unmarshal(data, &result); decodeErr != nil {
		return decodeErr
	}
	if result.Recipe != recipe || result.TotalCount != 3 || len(result.Embeddings) != 3 {
		return fmt.Errorf("wrong recipe/modality count: %s", data)
	}
	for i, modality := range []string{"", "image", "audio"} {
		output := result.Embeddings[i]
		if output.Modality != modality || output.Model != "multimodal" || output.Dimension != dimension || len(output.Vector) != dimension {
			return fmt.Errorf("unexpected %s representation metadata", modality)
		}
		norm := 0.0
		for _, value := range output.Vector {
			if math.IsNaN(value) || math.IsInf(value, 0) {
				return fmt.Errorf("non-finite %s vector", modality)
			}
			norm += value * value
		}
		if math.Abs(norm-1) > 1e-3 {
			return fmt.Errorf("%s vector norm squared = %g, expected 1 ± .001", modality, norm)
		}
	}
	// The audio route checks published representation semantics, not sound
	// recognition accuracy. These reference cosines use the original stereo PCM
	// and independently verified source-model text/audio vectors.
	wantCosine := map[string]float64{"nano": 0.04673499, "mini": 0.22175031}[recipe]
	cosine := 0.0
	for i, value := range result.Embeddings[0].Vector {
		cosine += value * result.Embeddings[2].Vector[i]
	}
	if math.Abs(cosine-wantCosine) > 1e-4 {
		return fmt.Errorf("text/audio reference cosine = %.8f; expected %.8f ± .0001", cosine, wantCosine)
	}
	for _, invalid := range []map[string]interface{}{
		{"texts": []string{"hello"}, "dimension": dimension / 2},
		{"texts": []string{"hello"}, "target_layer": 6},
		{"texts": []string{strings.Repeat("hello ", limit+16)}},
		{"audios": []string{"data:audio/wav;base64,YQ=="}},
	} {
		invalid["recipe"] = recipe
		invalid["model"] = "multimodal"
		status, _, data, err = p.post(ctx, p.apiURL+"/api/v1/diagnostics/embeddings", invalid)
		if err != nil {
			return err
		}
		if status != http.StatusBadRequest {
			return fmt.Errorf("invalid representation/input accepted: status %d: %s", status, data)
		}
	}
	if recipe == "mini" {
		status, _, data, err = p.post(ctx, p.apiURL+"/api/v1/diagnostics/embeddings", map[string]interface{}{"recipe": recipe, "model": "multimodal", "texts": []string{strings.Repeat("hello ", 1024)}})
		if err != nil {
			return err
		}
		if status != http.StatusOK {
			return fmt.Errorf("mini rejected its supported >512-token input: %d %s", status, data)
		}
	}
	return p.checkRoutes(ctx, recipe)
}

func (p omniProbe) checkInventory(ctx context.Context, recipe string, dimension, limit int) error {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, p.apiURL+"/api/v1/diagnostics/models?recipe="+recipe, nil)
	if err != nil {
		return err
	}
	response, err := p.client.Do(request)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	var inventory struct {
		Recipe   string `json:"recipe"`
		Bindings []struct {
			Name      string `json:"name"`
			Adapter   string `json:"adapter"`
			MaxTokens int    `json:"max_tokens"`
			Embedding *struct {
				Dimension  int      `json:"dimension"`
				Dimensions []int    `json:"dimensions"`
				Modalities []string `json:"modalities"`
				Audio      *struct {
					MaxSeconds  int `json:"max_seconds"`
					MaxChannels int `json:"max_channels"`
				} `json:"audio"`
			} `json:"embedding"`
		} `json:"bindings"`
	}
	if err := json.NewDecoder(io.LimitReader(response.Body, 1<<20)).Decode(&inventory); err != nil {
		return err
	}
	if response.StatusCode != http.StatusOK || inventory.Recipe != recipe {
		return fmt.Errorf("model inventory returned HTTP %d or wrong recipe %q", response.StatusCode, inventory.Recipe)
	}
	for _, model := range inventory.Bindings {
		if model.Name != "embedding" {
			continue
		}
		e := model.Embedding
		if model.Adapter != "vela_omni" || model.MaxTokens != limit || e == nil || e.Dimension != dimension || len(e.Dimensions) != 1 || e.Dimensions[0] != dimension || !slices.Equal(e.Modalities, []string{"text", "image", "audio"}) || e.Audio == nil || e.Audio.MaxSeconds != 30 || e.Audio.MaxChannels != 8 {
			return fmt.Errorf("loaded Omni capabilities do not match the published model: %+v", model)
		}
		return nil
	}
	return fmt.Errorf("loaded embedding binding is absent from inventory")
}

func (p omniProbe) checkRoutes(ctx context.Context, recipe string) error {
	for _, input := range []struct {
		content          interface{}
		decision, signal string
	}{
		{"A high pitched electronic tone.", "text-route", "exact-text"},
		{[]map[string]interface{}{{"type": "input_audio", "input_audio": map[string]string{"data": p.audio, "format": "wav"}}}, "audio-route", "tone"},
	} {
		status, headers, data, err := p.post(ctx, p.gatewayURL+"/v1/chat/completions", map[string]interface{}{"model": "vela-" + recipe, "messages": []map[string]interface{}{{"role": "user", "content": input.content}}, "max_tokens": 8})
		if err != nil {
			return err
		}
		if status != http.StatusOK || headers.Get("x-vsr-selected-decision") != input.decision {
			return fmt.Errorf("%s route: HTTP %d decision=%q: %s", input.decision, status, headers.Get("x-vsr-selected-decision"), data)
		}
		found := false
		for _, signal := range parseMatchedEmbeddingRules(headers.Get("x-vsr-matched-embeddings")) {
			found = found || signal == input.signal
		}
		if !found {
			return fmt.Errorf("route %s omitted actual %s embedding evidence", input.decision, input.signal)
		}
	}
	return nil
}

func (p omniProbe) post(ctx context.Context, url string, body interface{}) (int, http.Header, []byte, error) {
	raw, err := json.Marshal(body)
	if err != nil {
		return 0, nil, nil, err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(raw))
	if err != nil {
		return 0, nil, nil, err
	}
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("x-vsr-debug", "true")
	response, err := p.client.Do(request)
	if err != nil {
		return 0, nil, nil, err
	}
	defer response.Body.Close()
	data, err := io.ReadAll(io.LimitReader(response.Body, 2<<20))
	return response.StatusCode, response.Header, data, err
}

// One second of deterministic two-channel 44.1 kHz PCM. The two towers must
// derive their 16/48 kHz inputs independently from this original waveform.
func omniToneWAV() string {
	const rate = 44100
	const pcmBytes = rate * 2 * 2
	pcm := make([]byte, pcmBytes)
	for i := range rate {
		for channel, frequency := range []float64{440, 660} {
			value := int16(8192 * math.Sin(2*math.Pi*frequency*float64(i)/rate))
			offset := (i*2 + channel) * 2
			pcm[offset] = byte(value & 0xff)
			pcm[offset+1] = byte((value >> 8) & 0xff)
		}
	}
	wav := make([]byte, 44+len(pcm))
	copy(wav, "RIFF")
	binary.LittleEndian.PutUint32(wav[4:], 36+pcmBytes)
	copy(wav[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(wav[16:], 16)
	binary.LittleEndian.PutUint16(wav[20:], 1)
	binary.LittleEndian.PutUint16(wav[22:], 2)
	binary.LittleEndian.PutUint32(wav[24:], rate)
	binary.LittleEndian.PutUint32(wav[28:], rate*4)
	binary.LittleEndian.PutUint16(wav[32:], 4)
	binary.LittleEndian.PutUint16(wav[34:], 16)
	copy(wav[36:], "data")
	binary.LittleEndian.PutUint32(wav[40:], pcmBytes)
	copy(wav[44:], pcm)
	return base64.StdEncoding.EncodeToString(wav)
}
