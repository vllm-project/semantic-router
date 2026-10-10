package classification

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"slices"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

type audioSignalProvider struct {
	*embedding.FuncProvider
	calls   int
	request embedding.AudioRequest
}

func (p *audioSignalProvider) EmbedAudio(_ context.Context, request embedding.AudioRequest) ([]float32, error) {
	p.calls++
	p.request = request
	return []float32{1, 0}, nil
}

func TestAudioSignalReachesPreparedProviderThroughTypedEvaluation(t *testing.T) {
	wav := make([]byte, 48)
	copy(wav, "RIFF")
	binary.LittleEndian.PutUint32(wav[4:], 40)
	copy(wav[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(wav[16:], 16)
	binary.LittleEndian.PutUint16(wav[20:], 1)
	binary.LittleEndian.PutUint16(wav[22:], 2)
	binary.LittleEndian.PutUint32(wav[24:], 48000)
	binary.LittleEndian.PutUint32(wav[28:], 192000)
	binary.LittleEndian.PutUint16(wav[32:], 4)
	binary.LittleEndian.PutUint16(wav[34:], 16)
	copy(wav[36:], "data")
	binary.LittleEndian.PutUint32(wav[40:], 4)
	binary.LittleEndian.PutUint16(wav[44:], 16384)
	binary.LittleEndian.PutUint16(wav[46:], 49152)
	payload := "data:audio/wav;base64," + base64.StdEncoding.EncodeToString(wav)
	fp, _ := embedding.NewFuncProvider("synthetic-audio", 2, func(context.Context, string) ([]float32, error) { return []float32{1, 0}, nil })
	provider := &audioSignalProvider{FuncProvider: fp}
	rules := []config.EmbeddingRule{{Name: "audio", QueryModality: config.QueryModalityAudio, Candidates: []string{"a tone"}, SimilarityThreshold: .8, AggregationMethodConfiged: config.AggregationMethodMax}}
	ec, err := NewEmbeddingClassifierWithProvider(rules, config.HNSWConfig{PreloadEmbeddings: true}, provider)
	if err != nil {
		t.Fatal(err)
	}
	cfg := &config.RouterConfig{}
	cfg.EmbeddingRules = rules
	classifier := &Classifier{Config: cfg, keywordEmbeddingClassifier: ec}
	for _, tc := range []struct {
		name, audio     string
		matches, failed bool
	}{
		{"valid", payload, true, false},
		{"bad audio", "data:audio/wav;base64,YQ==", false, true},
		{"absent", "", false, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			results, err := classifier.EvaluateAllSignalsWithHeaders(SignalEvaluationInput{Audio: tc.audio, ForceEvaluateAll: true})
			if err != nil {
				t.Fatal(err)
			}
			if slices.Contains(results.MatchedEmbeddingRules, "audio") != tc.matches {
				t.Fatalf("matches=%v", results.MatchedEmbeddingRules)
			}
			if (results.SignalErrors["embedding:audio"] != "") != tc.failed {
				t.Fatalf("audio failure lost: %v", results.SignalErrors)
			}
		})
	}
	if provider.calls != 1 || provider.request.SampleRate != 48000 || provider.request.Channels != 2 || !slices.Equal(provider.request.PCM, []float32{.5, -.5}) {
		t.Fatalf("original PCM contract lost: calls=%d request=%+v", provider.calls, provider.request)
	}
}
