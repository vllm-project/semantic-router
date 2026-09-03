package testcases

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

// imageFilePNGBase64 is a 1x1 transparent PNG. The router sniffs the stored
// bytes and inlines them for the backend as a data URL carrying this exact
// base64 payload, so the provider-side assertion can match it verbatim.
const imageFilePNGBase64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="

const (
	imageFileVisionDecision = "input_modality_vision_decision"
	imageFileTextDecision   = "input_modality_text_decision"
)

func init() {
	pkgtestcases.Register("response-api-image-file-id", pkgtestcases.TestCase{
		Description: "POST /v1/responses with an uploaded vision file_id routes on the image and inlines it for the selected backend",
		Tags:        []string{"response-api", "functional", "files"},
		Fn:          testResponseAPIImageFileID,
	})
}

// testResponseAPIImageFileID pins the Response API image file_id contract
// end to end: an image uploaded through POST /v1/files (purpose=vision) and
// referenced by file_id in an input_image part must select the image decision
// AND arrive at the selected backend as inlined image bytes, while a file_id
// the router file store does not hold must fail with a client 400.
func testResponseAPIImageFileID(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing Response API: input_image file_id routing and backend delivery")
	}

	apiSession, err := fixtures.OpenRouterAPISession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer apiSession.Close()

	fileID, err := uploadVisionImage(ctx, apiSession)
	if err != nil {
		return fmt.Errorf("upload vision image: %w", err)
	}
	if opts.Verbose {
		fmt.Printf("[Test] Uploaded vision image: %s\n", fileID)
	}

	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()

	sessionID := fmt.Sprintf("response-api-image-file-id-%d", time.Now().UnixNano())
	if err := assertImageFileSelectsVisionDecision(ctx, session, fileID, sessionID); err != nil {
		return err
	}
	if err := verifyBackendReceivedInlinedImage(ctx, client, opts, sessionID, fileID); err != nil {
		return err
	}
	if err := assertTextOnlySelectsTextDecision(ctx, session); err != nil {
		return err
	}
	if err := assertUnknownImageFileRejected(ctx, session); err != nil {
		return err
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"file_id":             fileID,
			"image_decision":      imageFileVisionDecision,
			"text_decision":       imageFileTextDecision,
			"unknown_file_status": http.StatusBadRequest,
		})
	}
	return nil
}

// assertImageFileSelectsVisionDecision sends the uploaded image by file_id and
// asserts the image decision was selected.
func assertImageFileSelectsVisionDecision(
	ctx context.Context,
	session *fixtures.ServiceSession,
	fileID string,
	sessionID string,
) error {
	imageResp, err := postResponsesWithHeaders(ctx, session, map[string]any{
		"model": "MoM",
		"store": false,
		"input": []map[string]any{{
			"role": "user",
			"content": []map[string]any{
				{"type": "input_text", "text": "Describe the attached test image."},
				{"type": "input_image", "file_id": fileID},
			},
		}},
	}, map[string]string{"x-vsr-test-session-id": sessionID})
	if err != nil {
		return err
	}
	if imageResp.StatusCode != http.StatusOK {
		return fmt.Errorf("image request returned HTTP %d: %s", imageResp.StatusCode, truncateString(string(imageResp.Body), 500))
	}
	if decision := imageResp.Headers.Get("x-vsr-selected-decision"); decision != imageFileVisionDecision {
		return fmt.Errorf("image request selected decision %q, want %q", decision, imageFileVisionDecision)
	}
	return nil
}

// assertTextOnlySelectsTextDecision proves the image decision is driven by
// the image: a text-only request stays on the text decision.
func assertTextOnlySelectsTextDecision(ctx context.Context, session *fixtures.ServiceSession) error {
	textResp, err := postResponsesWithHeaders(ctx, session, map[string]any{
		"model": "MoM",
		"store": false,
		"input": "What is 2 + 2?",
	}, nil)
	if err != nil {
		return err
	}
	if textResp.StatusCode != http.StatusOK {
		return fmt.Errorf("text request returned HTTP %d: %s", textResp.StatusCode, truncateString(string(textResp.Body), 500))
	}
	if decision := textResp.Headers.Get("x-vsr-selected-decision"); decision != imageFileTextDecision {
		return fmt.Errorf("text request selected decision %q, want %q", decision, imageFileTextDecision)
	}
	return nil
}

// assertUnknownImageFileRejected pins the client 400 for a file_id the router
// file store does not hold, with the missing file named in the error.
func assertUnknownImageFileRejected(ctx context.Context, session *fixtures.ServiceSession) error {
	missingID := "file-e2e-does-not-exist"
	missingResp, err := postResponsesWithHeaders(ctx, session, map[string]any{
		"model": "MoM",
		"store": false,
		"input": []map[string]any{{
			"role": "user",
			"content": []map[string]any{
				{"type": "input_text", "text": "Describe the attached test image."},
				{"type": "input_image", "file_id": missingID},
			},
		}},
	}, nil)
	if err != nil {
		return err
	}
	if missingResp.StatusCode != http.StatusBadRequest {
		return fmt.Errorf("unknown file_id returned HTTP %d, want 400: %s", missingResp.StatusCode, truncateString(string(missingResp.Body), 500))
	}
	if !strings.Contains(string(missingResp.Body), missingID) {
		return fmt.Errorf("unknown file_id error does not name the file: %s", truncateString(string(missingResp.Body), 500))
	}
	return nil
}

// uploadVisionImage stores the test PNG in the router file store through the
// OpenAI-compatible Files API and returns the assigned file id.
func uploadVisionImage(ctx context.Context, session *fixtures.ServiceSession) (string, error) {
	imageBytes, err := base64.StdEncoding.DecodeString(imageFilePNGBase64)
	if err != nil {
		return "", fmt.Errorf("decode embedded test PNG: %w", err)
	}

	var form bytes.Buffer
	writer := multipart.NewWriter(&form)
	part, err := writer.CreateFormFile("file", "vsr-e2e-image.png")
	if err != nil {
		return "", err
	}
	if _, err := part.Write(imageBytes); err != nil {
		return "", err
	}
	if err := writer.WriteField("purpose", "vision"); err != nil {
		return "", err
	}
	if err := writer.Close(); err != nil {
		return "", err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, session.BaseURL()+"/v1/files", &form)
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", writer.FormDataContentType())
	resp, err := session.HTTPClient(30 * time.Second).Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("POST /v1/files returned HTTP %d: %s", resp.StatusCode, truncateString(string(body), 500))
	}

	var record struct {
		ID string `json:"id"`
	}
	if err := json.Unmarshal(body, &record); err != nil {
		return "", fmt.Errorf("decode /v1/files response: %w", err)
	}
	if record.ID == "" {
		return "", fmt.Errorf("POST /v1/files returned no file id: %s", truncateString(string(body), 500))
	}
	return record.ID, nil
}

// verifyBackendReceivedInlinedImage asserts that the request recorded by the
// mock backend is in Chat Completions form and carries the uploaded PNG as an
// inlined data URL, with the router-local file id gone.
func verifyBackendReceivedInlinedImage(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
	sessionID string,
	fileID string,
) error {
	providerSession, err := openProtocolCodecProviderSession(ctx, client, opts, "openai.chat.v1")
	if err != nil {
		return err
	}
	defer providerSession.Close()

	recorded, err := lastProviderSimulatorRequest(ctx, providerSession, sessionID)
	if err != nil {
		return err
	}
	var debug struct {
		Body map[string]json.RawMessage `json:"body"`
	}
	if err := json.Unmarshal(recorded, &debug); err != nil {
		return fmt.Errorf("decode recorded backend request: %w", err)
	}
	if _, found := debug.Body["messages"]; !found {
		return fmt.Errorf("backend request is not in Chat Completions form: %s", truncateString(string(recorded), 600))
	}
	inlined := "data:image/png;base64," + imageFilePNGBase64
	if !strings.Contains(string(recorded), inlined) {
		return fmt.Errorf("backend request lost the uploaded image bytes: %s", truncateString(string(recorded), 600))
	}
	if strings.Contains(string(recorded), fileID) {
		return fmt.Errorf("backend request leaked the router-local file id %q: %s", fileID, truncateString(string(recorded), 600))
	}
	return nil
}

// postResponsesWithHeaders sends POST /v1/responses through the gateway and
// returns the status, headers, and body for decision-header assertions.
func postResponsesWithHeaders(
	ctx context.Context,
	session *fixtures.ServiceSession,
	payload map[string]any,
	headers map[string]string,
) (*localChatCompletionResponse, error) {
	encoded, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, session.BaseURL()+"/v1/responses", bytes.NewReader(encoded))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	for key, value := range headers {
		req.Header.Set(key, value)
	}
	resp, err := session.HTTPClient(45 * time.Second).Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	return &localChatCompletionResponse{
		StatusCode: resp.StatusCode,
		Headers:    resp.Header.Clone(),
		Body:       body,
	}, nil
}
