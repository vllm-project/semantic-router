package services

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
)

func TestClassificationInputVariants(t *testing.T) {
	cases := []struct {
		body  string
		valid bool
	}{
		{`{"type":"text","text":"hello"}`, true},
		{`{"type":"image","image_url":"DATA:IMAGE/PNG;BASE64,YQ=="}`, true},
		{`{"type":"image","type":"text","text":"hello"}`, false},
		{`{"type":"text","text":"hello","IMAGE_URL":null}`, false},
		{`{"type":"image","TYPE":"text","text":"hello"}`, false},
		{`{"TYPE":"text","text":"hello"}`, false},
		{`null`, false},
		{`1`, false},
		{`"text"`, false},
		{`[]`, false},
		{`{}`, false},
		{`{"type":"audio","text":"hello"}`, false},
		{`{"type":"text","text":" "}`, false},
		{`{"type":"text","text":null}`, false},
		{`{"type":"text","text":"hello","image_url":null}`, false},
		{`{"type":"text","text":"hello","unknown":true}`, false},
		{`{"type":"text","text":42}`, false},
		{`{"type":"image","image_url":"https://example.com/image.png"}`, false},
		{`{"type":"image","image_url":"file:///etc/passwd"}`, false},
		{`{"type":"image","image_url":"./image.png"}`, false},
		{`{"type":"image","image_url":"data:image/svg+xml;base64,YQ=="}`, false},
		{`{"type":"image","image_url":"data:image/png;base64,!"}`, false},
		{`{"type":"image","image_url":"data:image/png;base64,"}`, false},
		{`{"type":"image","image_url":"data:image/png;base64,YQ==","text":null}`, false},
	}
	for _, tc := range cases {
		t.Run(tc.body, func(t *testing.T) {
			var input ClassificationInput
			if err := json.Unmarshal([]byte(tc.body), &input); err != nil {
				t.Fatal(err)
			}
			if err := input.Validate(); (err == nil) != tc.valid {
				t.Fatalf("valid=%v, err=%v", tc.valid, err)
			}
		})
	}
}

func TestClassificationInputMatchesLegacySignalInput(t *testing.T) {
	for _, body := range []string{`{"type":"text","text":"hello"}`, `{"type":"image","image_url":"DATA:IMAGE/PNG;BASE64,YQ=="}`} {
		var input ClassificationInput
		if err := json.Unmarshal([]byte(body), &input); err != nil {
			t.Fatal(err)
		}
		legacy, err := input.intentRequest(nil)
		if err != nil {
			t.Fatal(err)
		}
		want, err := legacy.resolveSignalInput()
		if err != nil {
			t.Fatal(err)
		}
		got, err := (IntentRequest{Input: &input}).resolveSignalInput()
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("typed input differs from existing path: %#v != %#v", got, want)
		}
	}
}

func TestTypedIntentRejectsUnsupportedImageBeforePlaceholder(t *testing.T) {
	image := "data:image/png;base64,YQ=="
	_, err := NewPlaceholderClassificationService().ClassifyIntent(context.Background(), IntentRequest{Input: &ClassificationInput{Type: "image", ImageURL: &image}})
	if !errors.Is(err, binding.ErrCapability) {
		t.Fatalf("got %v", err)
	}
}

func TestClassificationItemErrorsAreStableAndDoNotEchoPayload(t *testing.T) {
	for _, tc := range []struct {
		err  error
		code string
	}{
		{context.Canceled, "CANCELED"},
		{context.DeadlineExceeded, "DEADLINE_EXCEEDED"},
		{binding.ErrCapability, "UNSUPPORTED_INPUT"},
		{binding.ErrInputLimit, "INVALID_INPUT"},
		{errors.New("secret image bytes or remote traceback"), "CLASSIFICATION_ERROR"},
	} {
		got := ClassificationErrorForItem(tc.err)
		if got.Code != tc.code || got.Message == tc.err.Error() {
			t.Fatalf("unexpected error: %+v", got)
		}
	}
}
