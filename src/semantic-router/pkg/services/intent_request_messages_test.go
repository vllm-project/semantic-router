package services

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/imageurl"
)

func mustIntentImageDataURI(t *testing.T, mime string) string {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, 2, 2))
	img.Set(0, 0, color.RGBA{R: 32, G: 128, B: 255, A: 255})
	var encoded bytes.Buffer
	var err error
	switch mime {
	case "image/png":
		err = png.Encode(&encoded, img)
	case "image/jpeg":
		err = jpeg.Encode(&encoded, img, &jpeg.Options{Quality: 90})
	default:
		t.Fatalf("unsupported test MIME %q", mime)
	}
	require.NoError(t, err)
	return "data:" + mime + ";base64," + base64.StdEncoding.EncodeToString(encoded.Bytes())
}

func mustMessageContent(t *testing.T, value interface{}) json.RawMessage {
	t.Helper()
	data, err := json.Marshal(value)
	require.NoError(t, err)
	return data
}

func TestIntentRequestResolveSignalInput_UsesMessagesConversationHistory(t *testing.T) {
	req := IntentRequest{
		Messages: []IntentMessage{
			{
				Role:    "system",
				Content: mustMessageContent(t, "You are a careful tutor."),
			},
			{
				Role:    "user",
				Content: mustMessageContent(t, "Explain inflation vs recession in plain English."),
			},
			{
				Role:    "assistant",
				Content: mustMessageContent(t, "Inflation means prices rise over time."),
			},
			{
				Role: "user",
				Content: mustMessageContent(t, []map[string]string{
					{"type": "text", "text": "That was not clear."},
					{"type": "text", "text": "Explain inflation vs recession in plain English."},
				}),
			},
		},
	}

	input, err := req.resolveSignalInput()
	require.NoError(t, err)

	assert.Equal(t, "That was not clear. Explain inflation vs recession in plain English.", input.evaluationText)
	assert.Equal(t, input.evaluationText, input.currentUserText)
	assert.Equal(t, []string{"Explain inflation vs recession in plain English."}, input.priorUserMessages)
	assert.Equal(t, []string{"You are a careful tutor.", "Inflation means prices rise over time."}, input.nonUserMessages)
	assert.True(t, input.hasAssistantReply)
	assert.Equal(
		t,
		"You are a careful tutor. Inflation means prices rise over time. That was not clear. Explain inflation vs recession in plain English.",
		input.contextText,
	)
}

func TestIntentRequestResolveSignalInput_FallsBackToText(t *testing.T) {
	req := IntentRequest{Text: "Fallback single-turn request"}

	input, err := req.resolveSignalInput()
	require.NoError(t, err)

	assert.Equal(t, "Fallback single-turn request", input.evaluationText)
	assert.Equal(t, "Fallback single-turn request", input.contextText)
	assert.Equal(t, "Fallback single-turn request", input.currentUserText)
	assert.Empty(t, input.priorUserMessages)
	assert.Empty(t, input.nonUserMessages)
	assert.False(t, input.hasAssistantReply)
}

func TestIntentRequestResolveSignalInput_ExtractsImageFromCurrentUserTurn(t *testing.T) {
	dataURI := mustIntentImageDataURI(t, "image/png")
	req := IntentRequest{
		Messages: []IntentMessage{
			{
				Role: "user",
				Content: mustMessageContent(t, []map[string]interface{}{
					{"type": "text", "text": "What does this screenshot show?"},
					{"type": "image_url", "image_url": map[string]string{"url": dataURI}},
				}),
			},
		},
	}

	input, err := req.resolveSignalInput()
	require.NoError(t, err)

	assert.Equal(t, "What does this screenshot show?", input.evaluationText)
	assert.Equal(t, dataURI, input.imageURL)
}

func TestIntentRequestHasInlineImageInputMatchesUserImageWork(t *testing.T) {
	dataURI := mustIntentImageDataURI(t, "image/png")
	content := mustMessageContent(t, []map[string]interface{}{
		{"type": "image_url", "image_url": map[string]string{"url": dataURI}},
	})

	if !(IntentRequest{Messages: []IntentMessage{{Role: " user ", Content: content}}}).HasInlineImageInput() {
		t.Fatal("user inline image was not detected before admission")
	}
	if (IntentRequest{Messages: []IntentMessage{{Role: "system", Content: content}}}).HasInlineImageInput() {
		t.Fatal("ignored non-user image was treated as expensive user image work")
	}
	if (IntentRequest{Text: "text only"}).HasInlineImageInput() {
		t.Fatal("text-only request was treated as image work")
	}
}

func TestIntentRequestResolveSignalInput_AcceptsImageOnlyUserTurn(t *testing.T) {
	dataURI := mustIntentImageDataURI(t, "image/jpeg")
	req := IntentRequest{
		Messages: []IntentMessage{
			{
				Role: "user",
				Content: mustMessageContent(t, []map[string]interface{}{
					{"type": "image_url", "image_url": map[string]string{"url": dataURI}},
				}),
			},
		},
	}

	input, err := req.resolveSignalInput()
	require.NoError(t, err)

	assert.Empty(t, input.evaluationText)
	assert.Equal(t, dataURI, input.imageURL)
}

func TestIntentRequestResolveSignalInput_CanonicalizesUppercaseImageURL(t *testing.T) {
	// An uppercase-scheme data URI passes the safety gate, but the classifier
	// backend scans for ";base64," case-sensitively. The resolved imageURL must be
	// canonicalized (lowercase scheme/marker, payload preserved) so the image
	// signal actually fires on the classify/eval path instead of silently dropping.
	canonicalURI := mustIntentImageDataURI(t, "image/png")
	rawURI := strings.Replace(canonicalURI, "data:image/png;base64,", "DATA:IMAGE/PNG;BASE64,", 1)
	req := IntentRequest{
		Messages: []IntentMessage{
			{
				Role: "user",
				Content: mustMessageContent(t, []map[string]interface{}{
					{"type": "text", "text": "What does this screenshot show?"},
					{"type": "image_url", "image_url": map[string]string{"url": rawURI}},
				}),
			},
		},
	}

	input, err := req.resolveSignalInput()
	require.NoError(t, err)

	assert.Equal(t, "What does this screenshot show?", input.evaluationText)
	assert.Equal(t, canonicalURI, input.imageURL)
}

func TestIntentRequestResolveSignalInput_StringImageURLDoesNotPoisonText(t *testing.T) {
	// Responses API shape: image_url is a bare string, not a {"url": ...} object.
	// A string-valued part must not fail the whole content-parts unmarshal, which
	// would drop the text and regress a previously-classifiable request.
	dataURI := mustIntentImageDataURI(t, "image/png")
	req := IntentRequest{
		Messages: []IntentMessage{
			{
				Role: "user",
				Content: mustMessageContent(t, []map[string]interface{}{
					{"type": "input_text", "text": "hello world"},
					{"type": "input_image", "image_url": dataURI},
				}),
			},
		},
	}

	input, err := req.resolveSignalInput()
	require.NoError(t, err)

	assert.Equal(t, "hello world", input.evaluationText)
	assert.Equal(t, dataURI, input.imageURL)
}

func TestIntentRequestResolveSignalInput_MalformedImageURLDoesNotPoisonText(t *testing.T) {
	// A non-string, non-object image_url (e.g. a JSON number) must not fail the
	// whole content-parts unmarshal and drop the sibling text.
	req := IntentRequest{
		Messages: []IntentMessage{
			{
				Role: "user",
				Content: mustMessageContent(t, []map[string]interface{}{
					{"type": "input_text", "text": "keep this text"},
					{"type": "input_image", "image_url": 123},
				}),
			},
		},
	}

	input, err := req.resolveSignalInput()
	require.NoError(t, err)

	assert.Equal(t, "keep this text", input.evaluationText)
	assert.Empty(t, input.imageURL)
}

func TestIntentRequestResolveSignalInput_ImageOnlyTurnFallsBackToTopLevelText(t *testing.T) {
	// A safe-image-only message plus top-level text must still score the supplied
	// text; image safety (client-controlled) must not toggle whether it is scored.
	dataURI := mustIntentImageDataURI(t, "image/jpeg")
	req := IntentRequest{
		Text: "summarize my quarterly report",
		Messages: []IntentMessage{
			{
				Role: "user",
				Content: mustMessageContent(t, []map[string]interface{}{
					{"type": "image_url", "image_url": map[string]string{"url": dataURI}},
				}),
			},
		},
	}

	input, err := req.resolveSignalInput()
	require.NoError(t, err)

	assert.Equal(t, "summarize my quarterly report", input.evaluationText)
	assert.Equal(t, dataURI, input.imageURL)
}

func TestIntentRequestResolveSignalInput_ImageOnlyFollowUpKeepsPriorUserText(t *testing.T) {
	// An image-only follow-up turn must not rotate the real user question into
	// history and let the assistant reply be promoted into the scored slot.
	dataURI := mustIntentImageDataURI(t, "image/png")
	req := IntentRequest{
		Messages: []IntentMessage{
			{Role: "user", Content: mustMessageContent(t, "What is our refund policy?")},
			{Role: "assistant", Content: mustMessageContent(t, "Refunds are processed within 30 days.")},
			{
				Role: "user",
				Content: mustMessageContent(t, []map[string]interface{}{
					{"type": "image_url", "image_url": map[string]string{"url": dataURI}},
				}),
			},
		},
	}

	input, err := req.resolveSignalInput()
	require.NoError(t, err)

	assert.Equal(t, "What is our refund policy?", input.evaluationText)
	assert.Equal(t, dataURI, input.imageURL)
	assert.NotEqual(t, "Refunds are processed within 30 days.", input.evaluationText,
		"assistant reply must never be promoted into the scored evaluation text")
}

func TestIntentRequestResolveSignalInput_DropsUnsafeImageURL(t *testing.T) {
	req := IntentRequest{
		Messages: []IntentMessage{
			{
				Role: "user",
				Content: mustMessageContent(t, []map[string]interface{}{
					{"type": "text", "text": "Describe it."},
					{"type": "image_url", "image_url": map[string]string{"url": "https://example.com/cat.png"}},
				}),
			},
		},
	}

	input, err := req.resolveSignalInput()
	require.NoError(t, err)

	assert.Equal(t, "Describe it.", input.evaluationText)
	assert.Empty(t, input.imageURL, "non-data-URI image references must be rejected to prevent SSRF")
}

func TestIntentRequestResolveSignalInput_RejectsMalformedAllowlistedImages(t *testing.T) {
	validImage := mustIntentImageDataURI(t, "image/png")
	tests := []struct {
		name     string
		messages []IntentMessage
	}{
		{
			name: "image only malformed base64",
			messages: []IntentMessage{{
				Role: "user",
				Content: mustMessageContent(t, []map[string]interface{}{
					{"type": "image_url", "image_url": map[string]string{"url": "data:image/png;base64,!!!!"}},
				}),
			}},
		},
		{
			name: "sibling text does not hide malformed image",
			messages: []IntentMessage{{
				Role: "user",
				Content: mustMessageContent(t, []map[string]interface{}{
					{"type": "text", "text": "keep this text"},
					{"type": "image_url", "image_url": map[string]string{"url": "data:image/jpeg;base64,not-valid***"}},
				}),
			}},
		},
		{
			name: "empty base64 payload",
			messages: []IntentMessage{{
				Role: "user",
				Content: mustMessageContent(t, []map[string]interface{}{
					{"type": "image_url", "image_url": map[string]string{"url": "DATA:IMAGE/PNG;BASE64,   "}},
				}),
			}},
		},
		{
			name: "valid base64 containing non-image bytes",
			messages: []IntentMessage{{
				Role: "user",
				Content: mustMessageContent(t, []map[string]interface{}{
					{"type": "image_url", "image_url": map[string]string{"url": "data:image/png;base64,aGVsbG8="}},
				}),
			}},
		},
		{
			name: "valid GIF is unsupported by Candle",
			messages: []IntentMessage{{
				Role: "user",
				Content: mustMessageContent(t, []map[string]interface{}{
					{"type": "image_url", "image_url": map[string]string{"url": "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw=="}},
				}),
			}},
		},
		{
			name: "valid WebP is unsupported by Candle",
			messages: []IntentMessage{{
				Role: "user",
				Content: mustMessageContent(t, []map[string]interface{}{
					{"type": "image_url", "image_url": map[string]string{"url": "data:image/webp;base64,UklGRiIAAABXRUJQVlA4IBYAAAAwAQCdASoBAAEAAUAmJaQAA3AA/v89WAAAAA=="}},
				}),
			}},
		},
		{
			name: "later image part is validated",
			messages: []IntentMessage{{
				Role: "user",
				Content: mustMessageContent(t, []map[string]interface{}{
					{"type": "image_url", "image_url": map[string]string{"url": validImage}},
					{"type": "image_url", "image_url": map[string]string{"url": "data:image/webp;base64,UklGRiIAAABXRUJQVlA4IBYAAAAwAQCdASoBAAEAAUAmJaQAA3AA/v89WAAAAA=="}},
				}),
			}},
		},
		{
			name: "image in later user message is validated",
			messages: []IntentMessage{
				{
					Role: "user",
					Content: mustMessageContent(t, []map[string]interface{}{
						{"type": "text", "text": "first turn"},
						{"type": "image_url", "image_url": map[string]string{"url": validImage}},
					}),
				},
				{
					Role: "user",
					Content: mustMessageContent(t, []map[string]interface{}{
						{"type": "text", "text": "second turn"},
						{"type": "image_url", "image_url": map[string]string{"url": "data:image/png;base64,invalid!"}},
					}),
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := (IntentRequest{Messages: tt.messages}).resolveSignalInput()
			require.ErrorIs(t, err, ErrInvalidImageInput)
			assert.NotContains(t, err.Error(), "base64,")
		})
	}
}

func TestIntentRequestResolveSignalInput_BoundsImagesAcrossUserTurns(t *testing.T) {
	dataURI := mustIntentImageDataURI(t, "image/png")
	messages := make([]IntentMessage, 0, imageurl.MaxImagePartsPerRequest+1)
	for i := 0; i <= imageurl.MaxImagePartsPerRequest; i++ {
		messages = append(messages, IntentMessage{
			Role: "user",
			Content: mustMessageContent(t, []map[string]interface{}{
				{"type": "text", "text": "describe image"},
				{"type": "image_url", "image_url": map[string]string{"url": dataURI}},
			}),
		})
	}

	_, err := (IntentRequest{Messages: messages}).resolveSignalInput()
	require.ErrorIs(t, err, ErrInvalidImageInput)
}

func TestClassificationServiceClassifyIntentForEval_AcceptsMessagesWithoutText(t *testing.T) {
	service := &ClassificationService{classifier: nil}
	req := IntentRequest{
		Messages: []IntentMessage{
			{
				Role:    "user",
				Content: mustMessageContent(t, "Explain compound interest in one paragraph."),
			},
			{
				Role:    "assistant",
				Content: mustMessageContent(t, "Compound interest is interest on interest."),
			},
			{
				Role:    "user",
				Content: mustMessageContent(t, "That was not clear. Explain compound interest in one paragraph."),
			},
		},
	}

	resp, err := service.ClassifyIntentForEval(req)
	require.NoError(t, err)
	require.NotNil(t, resp)

	assert.Equal(t, "That was not clear. Explain compound interest in one paragraph.", resp.OriginalText)
	assert.NotNil(t, resp.Metrics)
}
