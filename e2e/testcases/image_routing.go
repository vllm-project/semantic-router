package testcases

import (
	"bytes"
	"compress/zlib"
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"hash/crc32"
	"io"
	"net/http"
	"strings"
	"time"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("systemone-image-input-routing", pkgtestcases.TestCase{
		Description: "A systemone question with input.image routes on what a request's image shows, and only that question receives the image",
		Tags:        []string{"signal-decision", "systemone", "multimodal", "routing"},
		Fn:          testSystemOneImageInputRouting,
	})
	pkgtestcases.Register("output-contract-image-generation", pkgtestcases.TestCase{
		Description: "A text-only request entering an image decision gets a generated image on the Responses API, and a Chat Completions client is refused before any model is called",
		Tags:        []string{"output-contract", "image_generation", "multimodal", "routing"},
		Fn:          testOutputContractImageGeneration,
	})
	pkgtestcases.Register("output-contract-text-and-image", pkgtestcases.TestCase{
		Description: "A text and image decision offers the hosted image_generation operation to one omni model without forcing it",
		Tags:        []string{"output-contract", "image_generation", "multimodal", "routing"},
		Fn:          testOutputContractTextAndImage,
	})
	pkgtestcases.Register("images-api-entry", pkgtestcases.TestCase{
		Description: "An Images API request routes on its prompt into the image decision and gets that model's image in the images dialect, and a prompt for a text route is refused before any model is called",
		Tags:        []string{"output-contract", "images-api", "multimodal", "routing"},
		Fn:          testImagesAPIEntry,
	})
}

const imageRoutingSessionHeader = "x-vsr-test-session-id"

type imageRoutingCase struct {
	name, text, imageMarker string
	wantDecision, wantModel string
}

// The image-routing profile's System One fixture reads a marker from the PNG
// text chunk of an image a question receives, and the Router sends images
// only to questions whose rules read them (input.image).
func testSystemOneImageInputRouting(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	localPort, stop, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stop()
	cases := []imageRoutingCase{
		{"an image of a receipt routes on what it shows", "What is in this picture?", "__SYSTEMONE_CHOICE__ receipt", "receipts", "openai/vision-model"},
		{"an image that shows nothing in particular takes the default route", "What is in this picture?", "", "default_decision", "openai/mock-model"},
		// The modality question does not read images, so the image's marker
		// cannot make it choose image; the image question has no such option.
		{"a question that does not read images never sees them", "Describe this picture.", "__SYSTEMONE_CHOICE__ image", "default_decision", "openai/mock-model"},
	}
	for _, tc := range cases {
		payload := map[string]any{
			"model": "vllm-sr/auto",
			"messages": []map[string]any{{"role": "user", "content": []map[string]any{
				{"type": "text", "text": tc.text},
				{"type": "image_url", "image_url": map[string]string{"url": pngDataURL(tc.imageMarker)}},
			}}},
		}
		response, err := postImageRoutingJSON(ctx, localPort, "/v1/chat/completions", payload, nil)
		if err != nil {
			return fmt.Errorf("%s: %w", tc.name, err)
		}
		if err := expectImageRoute(response, http.StatusOK, tc.wantDecision, tc.wantModel); err != nil {
			return fmt.Errorf("%s: %w", tc.name, err)
		}
	}
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]any{"cases": len(cases)})
	}
	return nil
}

// The fixture's modality question chooses image for a prompt that carries its
// choice marker, which selects the image decision.
func testOutputContractImageGeneration(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	localPort, stop, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stop()
	const prompt = "Draw a lighthouse at dusk. __SYSTEMONE_CHOICE__ image"
	session := fmt.Sprintf("output-contract-image-%d", time.Now().UnixNano())
	response, err := postImageRoutingJSON(ctx, localPort, "/v1/responses",
		map[string]any{"model": "vllm-sr/auto", "store": false, "input": prompt},
		map[string]string{imageRoutingSessionHeader: session})
	if err != nil {
		return err
	}
	if err = expectImageRoute(response, http.StatusOK, "image", "mock/image-generation"); err != nil {
		return err
	}
	if err = expectGeneratedImage(response.Body, false); err != nil {
		return err
	}
	recorded, err := lastImageRoutingBackendRequest(ctx, client, opts, session)
	if err != nil {
		return err
	}
	if recorded.Path != "/v1/images/generations" || !strings.Contains(string(recorded.Body["prompt"]), "Draw a lighthouse at dusk.") {
		return fmt.Errorf("the image model received %s %s, want the prompt on /v1/images/generations", recorded.Path, truncateString(string(recorded.Body["prompt"]), 200))
	}

	refused := session + "-chat"
	chat, err := postImageRoutingJSON(ctx, localPort, "/v1/chat/completions",
		map[string]any{"model": "vllm-sr/auto", "messages": []map[string]string{{"role": "user", "content": prompt}}},
		map[string]string{imageRoutingSessionHeader: refused})
	if err != nil {
		return err
	}
	if chat.StatusCode != http.StatusBadRequest || !strings.Contains(string(chat.Body), "Chat Completions API cannot return") {
		return fmt.Errorf("a Chat Completions client of the image decision got HTTP %d: %s", chat.StatusCode, truncateString(string(chat.Body), 400))
	}
	if _, err := lastImageRoutingBackendRequest(ctx, client, opts, refused); err == nil {
		return fmt.Errorf("the refused Chat Completions request reached a model")
	}
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]any{"decision": "image", "backend_path": recorded.Path})
	}
	return nil
}

func testOutputContractTextAndImage(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	localPort, stop, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stop()
	session := fmt.Sprintf("output-contract-text-image-%d", time.Now().UnixNano())
	response, err := postImageRoutingJSON(ctx, localPort, "/v1/responses",
		map[string]any{"model": "vllm-sr/auto", "store": false, "input": "Explain how a lighthouse works and draw one. __SYSTEMONE_CHOICE__ text_image"},
		map[string]string{imageRoutingSessionHeader: session})
	if err != nil {
		return err
	}
	if err = expectImageRoute(response, http.StatusOK, "text-and-image", "mock/omni-image"); err != nil {
		return err
	}
	if err = expectGeneratedImage(response.Body, true); err != nil {
		return err
	}
	recorded, err := lastImageRoutingBackendRequest(ctx, client, opts, session)
	if err != nil {
		return err
	}
	var tools []struct {
		Type string `json:"type"`
	}
	if recorded.Path != "/v1/responses" || json.Unmarshal(recorded.Body["tools"], &tools) != nil || len(tools) != 1 || tools[0].Type != "image_generation" {
		return fmt.Errorf("the omni model was not offered the image_generation operation on %s: %s", recorded.Path, truncateString(string(recorded.Body["tools"]), 300))
	}
	if choice := recorded.Body["tool_choice"]; len(choice) > 0 && string(choice) != `"auto"` {
		return fmt.Errorf("the omni model was forced to %s; text with an image leaves the call to the model", choice)
	}
	return nil
}

// The Images API entry routes on the prompt like the other APIs: the
// fixture's modality question chooses image for a prompt carrying its marker.
func testImagesAPIEntry(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	localPort, stop, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stop()
	session := fmt.Sprintf("images-api-%d", time.Now().UnixNano())
	response, err := postImageRoutingJSON(ctx, localPort, "/v1/images/generations",
		map[string]any{"model": "vllm-sr/auto", "prompt": "Draw a lighthouse at dusk. __SYSTEMONE_CHOICE__ image", "size": "512x512"},
		map[string]string{imageRoutingSessionHeader: session})
	if err != nil {
		return err
	}
	if err = expectImageRoute(response, http.StatusOK, "image", "mock/image-generation"); err != nil {
		return err
	}
	var images struct {
		Data []struct {
			B64JSON string `json:"b64_json"`
		} `json:"data"`
	}
	if err = json.Unmarshal(response.Body, &images); err != nil || len(images.Data) != 1 {
		return fmt.Errorf("the Images API client did not get one image: %s", truncateString(string(response.Body), 400))
	}
	if err = decedeImageResult(images.Data[0].B64JSON); err != nil {
		return err
	}
	recorded, err := lastImageRoutingBackendRequest(ctx, client, opts, session)
	if err != nil {
		return err
	}
	if recorded.Path != "/v1/images/generations" || !strings.Contains(string(recorded.Body["prompt"]), "Draw a lighthouse at dusk.") ||
		string(recorded.Body["size"]) != `"512x512"` {
		return fmt.Errorf("the image model received %s with prompt %s and size %s", recorded.Path,
			truncateString(string(recorded.Body["prompt"]), 200), recorded.Body["size"])
	}

	refused := session + "-text"
	text, err := postImageRoutingJSON(ctx, localPort, "/v1/images/generations",
		map[string]any{"model": "vllm-sr/auto", "prompt": "What is a lighthouse?"},
		map[string]string{imageRoutingSessionHeader: refused})
	if err != nil {
		return err
	}
	if text.StatusCode != http.StatusBadRequest || !strings.Contains(string(text.Body), "image_route_required") {
		return fmt.Errorf("an Images API prompt for a text route got HTTP %d: %s", text.StatusCode, truncateString(string(text.Body), 400))
	}
	if _, err := lastImageRoutingBackendRequest(ctx, client, opts, refused); err == nil {
		return fmt.Errorf("the refused Images API request reached a model")
	}
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]any{"decision": "image", "backend_path": recorded.Path})
	}
	return nil
}

func expectImageRoute(response *localChatCompletionResponse, status int, decision, model string) error {
	if response.StatusCode != status {
		return fmt.Errorf("HTTP %d, want %d: %s", response.StatusCode, status, truncateString(string(response.Body), 400))
	}
	if got := response.Headers.Get(vsrSelectedDecisionHeader); got != decision {
		return fmt.Errorf("x-vsr-selected-decision = %q, want %q", got, decision)
	}
	if got := response.Headers.Get("x-vsr-selected-model"); got != model {
		return fmt.Errorf("x-vsr-selected-model = %q, want %q", got, model)
	}
	return nil
}

// expectGeneratedImage requires a Responses body with a completed
// image_generation_call item whose result decodes; textAllowed admits the
// message an omni model may write beside it.
func expectGeneratedImage(body []byte, textAllowed bool) error {
	var decoded struct {
		Output []struct {
			Type   string `json:"type"`
			Status string `json:"status"`
			Result string `json:"result"`
		} `json:"output"`
	}
	if err := json.Unmarshal(body, &decoded); err != nil {
		return fmt.Errorf("decode the Responses body: %w", err)
	}
	images := 0
	for _, item := range decoded.Output {
		switch {
		case item.Type == "image_generation_call" && item.Status == "completed":
			if err := decedeImageResult(item.Result); err != nil {
				return err
			}
			images++
		case item.Type == "message" && textAllowed:
		default:
			return fmt.Errorf("unexpected output item %q: %s", item.Type, truncateString(string(body), 400))
		}
	}
	if images != 1 {
		return fmt.Errorf("%d generated images, want 1: %s", images, truncateString(string(body), 400))
	}
	return nil
}

type imageRoutingBackendRequest struct {
	Path string                     `json:"path"`
	Body map[string]json.RawMessage `json:"body"`
}

func lastImageRoutingBackendRequest(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions, session string) (*imageRoutingBackendRequest, error) {
	provider, err := fixtures.OpenServiceEndpointSession(ctx, client, opts, "default", "provider-mocker", "8000")
	if err != nil {
		return nil, err
	}
	defer provider.Close()
	recorded, err := lastProviderSimulatorRequest(ctx, provider, session)
	if err != nil {
		return nil, err
	}
	var request imageRoutingBackendRequest
	if err := json.Unmarshal(recorded, &request); err != nil {
		return nil, fmt.Errorf("decode the recorded backend request: %w", err)
	}
	return &request, nil
}

func postImageRoutingJSON(ctx context.Context, localPort, path string, payload any, headers map[string]string) (*localChatCompletionResponse, error) {
	encoded, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("http://localhost:%s%s", localPort, path), bytes.NewReader(encoded))
	if err != nil {
		return nil, err
	}
	request.Header.Set("Content-Type", "application/json")
	for key, value := range headers {
		request.Header.Set(key, value)
	}
	response, err := (&http.Client{Timeout: 45 * time.Second}).Do(request)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	body, err := io.ReadAll(response.Body)
	if err != nil {
		return nil, err
	}
	return &localChatCompletionResponse{StatusCode: response.StatusCode, Headers: response.Header.Clone(), Body: body}, nil
}

// pngDataURL is a 1x1 grayscale PNG data URL whose tEXt chunk carries text,
// or that has no text chunk when text is empty.
func pngDataURL(text string) string {
	var compressed bytes.Buffer
	writer := zlib.NewWriter(&compressed)
	_, _ = writer.Write([]byte{0, 0})
	_ = writer.Close()
	header := make([]byte, 13)
	binary.BigEndian.PutUint32(header[0:], 1)
	binary.BigEndian.PutUint32(header[4:], 1)
	header[8] = 8
	png := bytes.NewBufferString("\x89PNG\r\n\x1a\n")
	writePNGChunk(png, "IHDR", header)
	if text != "" {
		writePNGChunk(png, "tEXt", append([]byte("Comment\x00"), text...))
	}
	writePNGChunk(png, "IDAT", compressed.Bytes())
	writePNGChunk(png, "IEND", nil)
	return "data:image/png;base64," + base64.StdEncoding.EncodeToString(png.Bytes())
}

func writePNGChunk(png *bytes.Buffer, kind string, data []byte) {
	var length [4]byte
	binary.BigEndian.PutUint32(length[:], uint32(len(data))) // #nosec G115 -- chunks are a few bytes long
	png.Write(length[:])
	png.WriteString(kind)
	png.Write(data)
	var crc [4]byte
	binary.BigEndian.PutUint32(crc[:], crc32.ChecksumIEEE(append([]byte(kind), data...)))
	png.Write(crc[:])
}
