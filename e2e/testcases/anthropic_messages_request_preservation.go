package testcases

// Request-side field preservation tests for the Anthropic /v1/messages translation path.
//
// These tests assert that specific fields sent by the client reach the upstream
// backend (anthropic-shim) verbatim after the router's translation. Each test:
//  1. Sends a POST /v1/messages to the router with a tagged x-vsr-test-session-id.
//  2. Waits for a 200 OK.
//  3. Opens a second port-forward to the shim's debug service and queries
//     GET /debug/last-request to inspect what the shim received.
//  4. Asserts the specific field of interest.
//
// All eight tests require the anthropic-shim profile and are registered in
// AnthropicShimContract (e2e/pkg/testmatrix/testcases.go). Shared types and
// helpers live in anthropic_messages_request_preservation_helpers.go.
//
// IMPORTANT: The shim's /debug/last-request endpoint returns the request body
// AFTER the shim's own translations (join_system_array, join_tool_result_content).
// For scenarios 5, 6, 7, and 9 this makes no difference. For scenario 10,
// join_tool_result_content collapses the array to a newline-joined string before
// forwarding to llama-server; the assertion verifies the collapsed form
// (both parts present, in order) which is what actually reaches upstream.
//
// Header names are hardcoded as string literals here; the e2e module
// (e2e/go.mod) is intentionally decoupled from the router module
// (src/semantic-router/go.mod) and does not import pkg/headers.

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"time"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

func init() {
	pkgtestcases.Register("anthropic-messages-image-block-preserved", pkgtestcases.TestCase{
		Description: "Verify image content blocks survive router translation and reach the shim intact (anthropic-shim profile)",
		Tags:        []string{"anthropic", "preservation", "image"},
		Fn:          testAnthropicMessagesImageBlockPreserved,
	})
	pkgtestcases.Register("anthropic-messages-top-k-preserved", pkgtestcases.TestCase{
		Description: "Verify top_k reaches upstream and no lossiness warning is emitted when backend is Anthropic-shaped (anthropic-shim profile)",
		Tags:        []string{"anthropic", "preservation", "top_k"},
		Fn:          testAnthropicMessagesTopKPreserved,
	})
	pkgtestcases.Register("anthropic-messages-metadata-user-id-preserved", pkgtestcases.TestCase{
		Description: "Verify metadata.user_id reaches the shim verbatim through router translation (anthropic-shim profile)",
		Tags:        []string{"anthropic", "preservation", "metadata"},
		Fn:          testAnthropicMessagesMetadataUserIDPreserved,
	})
	pkgtestcases.Register("anthropic-messages-tool-result-is-error-preserved", pkgtestcases.TestCase{
		Description: "Verify tool_result.is_error=true is preserved through router translation (anthropic-shim profile)",
		Tags:        []string{"anthropic", "preservation", "tool_result"},
		Fn:          testAnthropicMessagesToolResultIsErrorPreserved,
	})
	pkgtestcases.Register("anthropic-messages-tool-result-array-content-preserved", pkgtestcases.TestCase{
		Description: "Verify multi-block tool_result.content array is forwarded to upstream with both parts present (anthropic-shim profile)",
		Tags:        []string{"anthropic", "preservation", "tool_result", "array"},
		Fn:          testAnthropicMessagesToolResultArrayContentPreserved,
	})
	pkgtestcases.Register("anthropic-messages-stop-sequences-preserved", pkgtestcases.TestCase{
		Description: "Verify stop_sequences survives the Anthropic→IR→Anthropic round-trip and reaches the shim verbatim (anthropic-shim profile)",
		Tags:        []string{"anthropic", "preservation", "stop_sequences"},
		Fn:          testAnthropicMessagesStopSequencesPreserved,
	})
	pkgtestcases.Register("anthropic-messages-temperature-preserved", pkgtestcases.TestCase{
		Description: "Verify temperature survives the Anthropic→IR→Anthropic round-trip and reaches the shim verbatim (anthropic-shim profile)",
		Tags:        []string{"anthropic", "preservation", "sampling"},
		Fn:          testAnthropicMessagesTemperaturePreserved,
	})
	pkgtestcases.Register("anthropic-messages-top-p-preserved", pkgtestcases.TestCase{
		Description: "Verify top_p survives the Anthropic→IR→Anthropic round-trip and reaches the shim verbatim (anthropic-shim profile)",
		Tags:        []string{"anthropic", "preservation", "sampling"},
		Fn:          testAnthropicMessagesTopPPreserved,
	})
}

// ── Test: image block survival (scenario 5) ───────────────────────────────────

// testAnthropicMessagesImageBlockPreserved verifies scenario 5:
// a user message carrying a base64 image block is forwarded to the shim with
// the image block intact (type, source.type, source.media_type, source.data
// all present and matching the original values).
//
// Requires the anthropic-shim profile.
func testAnthropicMessagesImageBlockPreserved(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Anthropic] Testing image block preservation through /v1/messages translation")
	}

	routerPort, stopRouter, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopRouter()

	shimPort, stopShim, err := openShimSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopShim()

	sessionID := fmt.Sprintf("preservation-image-%d", time.Now().UnixNano())

	body := preservationRequestBody{
		Model:     "MoM",
		MaxTokens: 32,
		Messages: []preservationMessage{
			{
				Role: "user",
				Content: []preservationContentBlock{
					{
						Type: "image",
						Source: imageSource{
							Type:      "base64",
							MediaType: "image/png",
							Data:      tinyPNG1x1,
						},
					},
					{
						Type: "text",
						Text: "What colour is this image?",
					},
				},
			},
		},
	}

	resp, err := postPreservationRequest(ctx, body, sessionID, routerPort)
	if err != nil {
		return fmt.Errorf("router request: %w", err)
	}
	defer resp.Body.Close()
	rawBody, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("expected 200 from router, got %d: %s", resp.StatusCode, truncateString(string(rawBody), 200))
	}

	debug, err := fetchShimDebugBody(ctx, shimPort, sessionID)
	if err != nil {
		return fmt.Errorf("shim debug: %w", err)
	}

	// Navigate: body.messages[0].content[0] must be the image block.
	imageBlock, err := extractFirstContentBlock(debug.Body)
	if err != nil {
		return fmt.Errorf("extract image block from shim body: %w", err)
	}

	got, err := readImageBlock(imageBlock)
	if err != nil {
		return err
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"block_type":        "image",
			"source_type":       got.Type,
			"source_media_type": got.MediaType,
			"source_data_len":   len(got.Data),
		})
	}

	want := imageSource{Type: "base64", MediaType: "image/png", Data: tinyPNG1x1}
	return assertImageSourceEqual(want, got)
}

// ── Test: top_k reaching upstream (scenario 6) ───────────────────────────────

// testAnthropicMessagesTopKPreserved verifies scenario 6:
// top_k sent by the client reaches the shim with the correct value AND the
// response does NOT carry x-vsr-lossiness-warnings, since the backend is
// Anthropic-shaped and top_k should be forwarded without loss.
//
// Requires the anthropic-shim profile.
func testAnthropicMessagesTopKPreserved(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Anthropic] Testing top_k preservation through /v1/messages translation")
	}

	routerPort, stopRouter, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopRouter()

	shimPort, stopShim, err := openShimSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopShim()

	sessionID := fmt.Sprintf("preservation-topk-%d", time.Now().UnixNano())

	body := preservationRequestBody{
		Model:     "MoM",
		MaxTokens: 32,
		TopK:      40,
		Messages: []preservationMessage{
			{
				Role: "user",
				Content: []preservationContentBlock{
					{Type: "text", Text: "Say one word."},
				},
			},
		},
	}

	resp, err := postPreservationRequest(ctx, body, sessionID, routerPort)
	if err != nil {
		return fmt.Errorf("router request: %w", err)
	}
	defer resp.Body.Close()
	rawBody, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("expected 200 from router, got %d: %s", resp.StatusCode, truncateString(string(rawBody), 200))
	}

	// x-vsr-lossiness-warnings must be absent for Anthropic-shaped backends.
	// When the backend is OpenAI-shaped, top_k cannot be forwarded and the router
	// emits "top_k_drop_on_openai_backend". With the anthropic-shim the field
	// should reach upstream without loss, so no warning is expected.
	lossiness := resp.Header.Get("x-vsr-lossiness-warnings")

	debug, err := fetchShimDebugBody(ctx, shimPort, sessionID)
	if err != nil {
		return fmt.Errorf("shim debug: %w", err)
	}

	// top_k is preserved in the body forwarded to upstream.
	topKRaw, ok := debug.Body["top_k"]
	if !ok {
		return fmt.Errorf("expected top_k field in shim body, not present")
	}
	// JSON numbers unmarshal as float64 in interface{} maps.
	topKVal, ok := topKRaw.(float64)
	if !ok {
		return fmt.Errorf("expected top_k to be a number, got %T", topKRaw)
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"top_k_in_shim":    int(topKVal),
			"lossiness_header": lossiness,
		})
	}

	if int(topKVal) != 40 {
		return fmt.Errorf("expected top_k=40 in shim body, got %d", int(topKVal))
	}
	if lossiness != "" {
		return fmt.Errorf("expected no x-vsr-lossiness-warnings for Anthropic-shaped backend, got %q", lossiness)
	}

	return nil
}

// ── Test: metadata.user_id reaching upstream (scenario 7) ────────────────────

// testAnthropicMessagesMetadataUserIDPreserved verifies scenario 7:
// metadata.user_id sent by the client reaches the shim verbatim after router
// translation.
//
// Requires the anthropic-shim profile.
func testAnthropicMessagesMetadataUserIDPreserved(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Anthropic] Testing metadata.user_id preservation through /v1/messages translation")
	}

	routerPort, stopRouter, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopRouter()

	shimPort, stopShim, err := openShimSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopShim()

	sessionID := fmt.Sprintf("preservation-metadata-%d", time.Now().UnixNano())
	const wantUserID = "test-user-12345"

	body := preservationRequestBody{
		Model:     "MoM",
		MaxTokens: 32,
		Metadata:  map[string]interface{}{"user_id": wantUserID},
		Messages: []preservationMessage{
			{
				Role: "user",
				Content: []preservationContentBlock{
					{Type: "text", Text: "Say one word."},
				},
			},
		},
	}

	resp, err := postPreservationRequest(ctx, body, sessionID, routerPort)
	if err != nil {
		return fmt.Errorf("router request: %w", err)
	}
	defer resp.Body.Close()
	rawBody, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("expected 200 from router, got %d: %s", resp.StatusCode, truncateString(string(rawBody), 200))
	}

	debug, err := fetchShimDebugBody(ctx, shimPort, sessionID)
	if err != nil {
		return fmt.Errorf("shim debug: %w", err)
	}

	metaRaw, ok := debug.Body["metadata"]
	if !ok {
		return fmt.Errorf("expected metadata field in shim body, not present")
	}
	meta, ok := metaRaw.(map[string]interface{})
	if !ok {
		return fmt.Errorf("expected metadata to be an object, got %T", metaRaw)
	}
	gotUserID, _ := meta["user_id"].(string)

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"metadata_user_id": gotUserID,
		})
	}

	if gotUserID != wantUserID {
		return fmt.Errorf("expected metadata.user_id=%q in shim body, got %q", wantUserID, gotUserID)
	}

	return nil
}

// ── Test: tool_result.is_error preservation (scenario 9) ─────────────────────

// testAnthropicMessagesToolResultIsErrorPreserved verifies scenario 9:
// is_error=true on a tool_result block reaches the shim after router translation.
// The request has an assistant turn (to satisfy the alternating-turns contract)
// followed by a user turn with a tool_result block carrying is_error=true.
//
// Requires the anthropic-shim profile.
func testAnthropicMessagesToolResultIsErrorPreserved(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Anthropic] Testing tool_result.is_error=true preservation through /v1/messages translation")
	}

	routerPort, stopRouter, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopRouter()

	shimPort, stopShim, err := openShimSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopShim()

	sessionID := fmt.Sprintf("preservation-iserror-%d", time.Now().UnixNano())

	body := preservationRequestBody{
		Model:     "MoM",
		MaxTokens: 32,
		Messages: []preservationMessage{
			{
				Role: "user",
				Content: []preservationContentBlock{
					{Type: "text", Text: "Call the tool."},
				},
			},
			{
				Role: "assistant",
				Content: []preservationContentBlock{
					{
						Type:      "tool_use",
						ToolUseID: "tu_x",
						Text:      "do_something",
					},
				},
			},
			{
				Role: "user",
				Content: []preservationContentBlock{
					{
						Type:      "tool_result",
						ToolUseID: "tu_x",
						IsError:   true,
						Content:   "Error!",
					},
				},
			},
		},
	}

	resp, err := postPreservationRequest(ctx, body, sessionID, routerPort)
	if err != nil {
		return fmt.Errorf("router request: %w", err)
	}
	defer resp.Body.Close()
	rawBody, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("expected 200 from router, got %d: %s", resp.StatusCode, truncateString(string(rawBody), 200))
	}

	debug, err := fetchShimDebugBody(ctx, shimPort, sessionID)
	if err != nil {
		return fmt.Errorf("shim debug: %w", err)
	}

	// Navigate: body.messages[2].content[0] is the tool_result block.
	isError, err := extractToolResultIsError(debug.Body)
	if err != nil {
		return fmt.Errorf("extract is_error from shim body: %w", err)
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"tool_result_is_error": isError,
		})
	}

	if !isError {
		return fmt.Errorf("expected tool_result.is_error=true in shim body, got false")
	}

	return nil
}

// ── Test: tool_result array content preservation (scenario 10) ───────────────

// testAnthropicMessagesToolResultArrayContentPreserved verifies scenario 10:
// a tool_result block with multi-part array content (two text blocks) is
// forwarded to upstream. The shim's join_tool_result_content collapses the
// array into a newline-joined string before recording and forwarding, so the
// assertion checks the joined string — "part 1\npart 2" — which is what
// actually reaches llama-server. Both parts must be present, in order.
//
// Requires the anthropic-shim profile.
func testAnthropicMessagesToolResultArrayContentPreserved(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Anthropic] Testing tool_result array content preservation through /v1/messages translation")
	}

	routerPort, stopRouter, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopRouter()

	shimPort, stopShim, err := openShimSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopShim()

	sessionID := fmt.Sprintf("preservation-trarr-%d", time.Now().UnixNano())

	// The tool_result.content is a two-element array of text blocks.
	// The shim collapses this to "part 1\npart 2" via join_tool_result_content
	// before forwarding to llama-server. We assert on the collapsed form.
	body := preservationRequestBody{
		Model:     "MoM",
		MaxTokens: 32,
		Messages: []preservationMessage{
			{
				Role: "user",
				Content: []preservationContentBlock{
					{Type: "text", Text: "Call the tool."},
				},
			},
			{
				Role: "assistant",
				Content: []preservationContentBlock{
					{
						Type:      "tool_use",
						ToolUseID: "tu_y",
						Text:      "fetch_data",
					},
				},
			},
			{
				Role: "user",
				Content: []preservationContentBlock{
					{
						Type:      "tool_result",
						ToolUseID: "tu_y",
						Content: []map[string]string{
							{"type": "text", "text": "part 1"},
							{"type": "text", "text": "part 2"},
						},
					},
				},
			},
		},
	}

	resp, err := postPreservationRequest(ctx, body, sessionID, routerPort)
	if err != nil {
		return fmt.Errorf("router request: %w", err)
	}
	defer resp.Body.Close()
	rawBody, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("expected 200 from router, got %d: %s", resp.StatusCode, truncateString(string(rawBody), 200))
	}

	debug, err := fetchShimDebugBody(ctx, shimPort, sessionID)
	if err != nil {
		return fmt.Errorf("shim debug: %w", err)
	}

	// The shim's join_tool_result_content collapses ["part 1", "part 2"] to
	// "part 1\npart 2". Both original parts must appear in the forwarded content.
	content, err := extractToolResultContent(debug.Body)
	if err != nil {
		return fmt.Errorf("extract tool_result.content from shim body: %w", err)
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"tool_result_content": content,
		})
	}

	const wantJoined = "part 1\npart 2"
	if content != wantJoined {
		return fmt.Errorf("expected tool_result.content=%q after shim join, got %q", wantJoined, content)
	}

	return nil
}

// ── Test: stop_sequences round-trip preservation ─────────────────────────────

// testAnthropicMessagesStopSequencesPreserved verifies that stop_sequences
// survives the full Anthropic→IR→Anthropic round-trip. Unlike top_k (which
// rides the AnthropicPassthrough sidecar), stop_sequences is mapped into the
// OpenAI IR's Stop field on inbound (applyStopSequences) and mapped back out to
// the Anthropic StopSequences field on outbound (applySampling). A regression
// in either re-mapping would drop or corrupt the field; the shim forwards it
// verbatim, so /debug/last-request reflects exactly what reached upstream.
//
// Requires the anthropic-shim profile.
func testAnthropicMessagesStopSequencesPreserved(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Anthropic] Testing stop_sequences preservation through /v1/messages translation")
	}

	routerPort, stopRouter, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopRouter()

	shimPort, stopShim, err := openShimSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopShim()

	sessionID := fmt.Sprintf("preservation-stopseq-%d", time.Now().UnixNano())
	wantSequences := []string{"END", "STOP"}

	body := preservationRequestBody{
		Model:         "MoM",
		MaxTokens:     32,
		StopSequences: wantSequences,
		Messages: []preservationMessage{
			{
				Role: "user",
				Content: []preservationContentBlock{
					{Type: "text", Text: "Say one word."},
				},
			},
		},
	}

	resp, err := postPreservationRequest(ctx, body, sessionID, routerPort)
	if err != nil {
		return fmt.Errorf("router request: %w", err)
	}
	defer resp.Body.Close()
	rawBody, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("expected 200 from router, got %d: %s", resp.StatusCode, truncateString(string(rawBody), 200))
	}

	debug, err := fetchShimDebugBody(ctx, shimPort, sessionID)
	if err != nil {
		return fmt.Errorf("shim debug: %w", err)
	}

	got, err := extractStopSequences(debug.Body)
	if err != nil {
		return fmt.Errorf("extract stop_sequences from shim body: %w", err)
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"stop_sequences": got,
		})
	}

	if !equalStringSlices(got, wantSequences) {
		return fmt.Errorf("expected stop_sequences=%v in shim body, got %v", wantSequences, got)
	}

	return nil
}

// ── Test: temperature round-trip preservation ────────────────────────────────

// testAnthropicMessagesTemperaturePreserved verifies that temperature
// survives the Anthropic→IR→Anthropic round-trip. It is mapped into the
// OpenAI IR's Temperature field on inbound (applyNumericSampling) and back to
// the Anthropic Temperature field on outbound (applySampling), with no scaling
// in either direction. A regression in the Valid()-guarded re-mapping would
// silently drop the field. 0.7 is valid in both the OpenAI (0-2) and Anthropic
// (0-1) ranges, so the value reaching the shim must match exactly.
//
// Requires the anthropic-shim profile.
func testAnthropicMessagesTemperaturePreserved(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	return runSamplingFieldPreservation(ctx, client, opts, samplingFieldCase{
		label:     "temperature",
		sessionID: "preservation-temp",
		field:     "temperature",
		want:      0.7,
		apply:     func(b *preservationRequestBody, v float64) { b.Temperature = &v },
	})
}

// ── Test: top_p round-trip preservation ──────────────────────────────────────

// testAnthropicMessagesTopPPreserved verifies that top_p survives the
// Anthropic→IR→Anthropic round-trip (applyNumericSampling inbound, applySampling
// outbound, no scaling). A regression in the Valid()-guarded re-mapping would
// silently drop the field.
//
// Requires the anthropic-shim profile.
func testAnthropicMessagesTopPPreserved(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	return runSamplingFieldPreservation(ctx, client, opts, samplingFieldCase{
		label:     "top_p",
		sessionID: "preservation-topp",
		field:     "top_p",
		want:      0.9,
		apply:     func(b *preservationRequestBody, v float64) { b.TopP = &v },
	})
}
