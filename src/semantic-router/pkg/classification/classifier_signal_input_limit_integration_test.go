//go:build !windows && cgo

package classification_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	candle "github.com/vllm-project/semantic-router/candle-binding"
	ort "github.com/vllm-project/semantic-router/onnx-binding/instance"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/apiserver"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

func TestSignalInputLimitClassificationEvalAndPreview(t *testing.T) {
	privateMessage := "input_limit at /private/example.bin: synthetic backend detail"
	cases := []struct {
		name    string
		err     error
		limited bool
	}{
		{"candle", fmt.Errorf("%w: %w", binding.ErrInputLimit, &candle.InstanceError{Code: "input_limit", Message: privateMessage}), true},
		{"ort", fmt.Errorf("%w: %w", binding.ErrInputLimit, &ort.Error{Kind: "input_limit", Message: privateMessage}), true},
		{"untyped text", errors.New(privateMessage), false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			classifier, err := classification.NewSignalErrorTestClassifier(tc.err)
			require.NoError(t, err)
			service := services.NewClassificationService(classifier, classifier.Config)
			expected := map[string]string{"jailbreak:guard": "jailbreak_evaluation_failed", "pii:private": "pii_evaluation_failed"}
			if tc.limited {
				for key := range expected {
					expected[key] = "input_limit"
				}
			}
			response, err := service.ClassifyIntentForEval(context.Background(), services.IntentRequest{Text: "neutral request", Options: &services.IntentOptions{Trace: true}})
			require.ErrorIs(t, err, decision.ErrDecisionUnresolved)
			require.NotNil(t, response)
			require.Equal(t, expected, response.SignalErrors)
			require.Equal(t, "fail_request", response.AppliedUnknownPolicies["guarded"])
			require.Equal(t, "fail_request", response.AppliedUnknownPolicies["private"])

			// Start the real management routes with a borrowed, model-free classification service.
			listener, err := net.Listen("tcp", "127.0.0.1:0")
			require.NoError(t, err)
			port := listener.Addr().(*net.TCPAddr).Port
			require.NoError(t, listener.Close())
			registry := routerruntime.NewRegistry(classifier.Config)
			registry.SetClassificationService(service)
			server, err := apiserver.StartWithOptions(apiserver.InitOptions{Port: port, BindAddress: "127.0.0.1", AuthMode: "disabled", RuntimeRegistry: registry})
			require.NoError(t, err)
			t.Cleanup(func() {
				ctx, cancel := context.WithTimeout(context.Background(), time.Second)
				defer cancel()
				require.NoError(t, server.Shutdown(ctx))
			})
			client := &http.Client{Timeout: time.Second}
			request, err := http.NewRequest(http.MethodPost, fmt.Sprintf("http://127.0.0.1:%d/api/v1/routing/preview?trace=true", port), bytes.NewBufferString(`{"text":"neutral request"}`))
			require.NoError(t, err)
			request.Header.Set("Content-Type", "application/json")
			actual, err := client.Do(request)
			require.NoError(t, err)
			defer actual.Body.Close()
			raw, err := io.ReadAll(actual.Body)
			require.NoError(t, err)
			require.Equal(t, http.StatusServiceUnavailable, actual.StatusCode, string(raw))
			var got services.EvalResponse
			require.NoError(t, json.Unmarshal(raw, &got))
			require.Equal(t, expected, got.SignalErrors)
			require.Equal(t, response.AppliedUnknownPolicies, got.AppliedUnknownPolicies)
			require.NotEmpty(t, got.DecisionError)
			require.Empty(t, got.SelectedModel)
			require.NotEqual(t, services.EvalSelectionSelected, got.SelectionStatus)
			require.NotContains(t, string(raw), "/private/example.bin")
			require.NotContains(t, string(raw), "synthetic backend detail")
		})
	}
}
