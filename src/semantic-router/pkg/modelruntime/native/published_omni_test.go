package native

import (
	"bytes"
	"context"
	"errors"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
)

// TestPublishedOmniModels qualifies the prepared ORT CPU releases. The 1024-token
// Mini probe exercises its longer-input path; it does not qualify 32K inference.
func TestPublishedOmniModels(t *testing.T) {
	paths := []string{os.Getenv("VLLM_SR_OMNI_NANO_MODEL"), os.Getenv("VLLM_SR_OMNI_MINI_MODEL")}
	for _, path := range paths {
		if path == "" {
			if os.Getenv("REQUIRE_OMNI_TESTS") == "1" {
				t.Fatal("required Nano and Mini artifact paths are absent")
			}
			t.Skip("published Omni requires explicit Nano and Mini artifacts")
		}
	}
	if provider := os.Getenv("VLLM_SR_MODEL_TEST_PROVIDER"); provider != "" && provider != "ort" {
		t.Fatal("Omni runtime inventory requires ORT")
	}
	if device := os.Getenv("VLLM_SR_MODEL_TEST_DEVICE"); device != "" && device != "cpu" {
		t.Fatal("Omni runtime inventory qualifies CPU only")
	}
	runtime := New(nil)
	parent := t
	var nano *EmbeddingProvider
	var first []float32
	for index, variant := range []string{"nano", "mini"} {
		t.Run(variant, func(t *testing.T) {
			path, err := filepath.Abs(paths[index])
			if err != nil {
				t.Fatal(err)
			}
			dimension, limit := 384, 512
			if variant == "mini" {
				dimension, limit = 768, 32768
			}
			spec := config.ResolvedModelBinding{
				Recipe: config.RecipeName(variant), Name: "embedding",
				Binding:    config.ModelBinding{Deployment: variant, Adapter: "vela_omni", Contract: "embedding.v1"},
				Deployment: config.ModelDeployment{Artifact: path, Provider: "ort", Device: "cpu", Precision: "native", Input: config.ModelInputBudget{MaxTokens: limit, Overflow: "reject"}},
			}
			model, err := runtime.Embedding(context.Background(), spec, 0, 0)
			if err != nil {
				t.Fatal(err)
			}
			// Keep both recipe-owned providers alive until the final Nano revisit.
			parent.Cleanup(func() {
				if err := model.Close(); err != nil {
					parent.Error(err)
				}
			})
			vector := checkPublishedOmni(t, model, variant, dimension, limit)
			if variant == "nano" {
				nano, first = model, vector
			}
		})
	}
	t.Run("nano-revisit", func(t *testing.T) {
		if nano == nil {
			t.Fatal("Nano was not prepared")
		}
		vector, err := nano.Embed(context.Background(), "A high pitched electronic tone.")
		if err != nil {
			t.Fatal(err)
		}
		if len(vector) != len(first) {
			t.Fatal("Mini changed Nano's representation dimension")
		}
		for i := range vector {
			if math.Abs(float64(vector[i]-first[i])) > 1e-6 {
				t.Fatal("Mini changed the still-owned Nano representation")
			}
		}
	})
}

func checkPublishedOmni(t *testing.T, model *EmbeddingProvider, variant string, dimension, limit int) []float32 {
	t.Helper()
	ctx := context.Background()
	info := model.EmbeddingInfo()
	if info.Backend != "ort" || info.Dimension != dimension || model.Dimension() != dimension || info.MaxTokens != limit || !slices.Equal(info.Dimensions, []int{dimension}) || !slices.Equal(info.Modalities, []string{"text", "image", "audio"}) || info.Audio == nil || info.Audio.MaxSeconds != 30 || info.Audio.MaxChannels != 8 {
		t.Fatalf("unexpected published capabilities: %+v", info)
	}
	text, err := model.Embed(ctx, "A high pitched electronic tone.")
	if err != nil {
		t.Fatal(err)
	}
	picture := image.NewRGBA(image.Rect(0, 0, 32, 32))
	for y := range 32 {
		for x := range 32 {
			picture.Set(x, y, color.RGBA{uint8(x * 8), uint8(y * 8), 128, 255})
		}
	}
	var encoded bytes.Buffer
	if encodeErr := png.Encode(&encoded, picture); encodeErr != nil {
		t.Fatal(encodeErr)
	}
	visual, err := model.EmbedImage(ctx, encoded.Bytes(), dimension)
	if err != nil {
		t.Fatal(err)
	}
	audio, err := model.EmbedAudio(ctx, publishedOmniTone())
	if err != nil {
		t.Fatal(err)
	}
	for modality, vector := range map[string][]float32{"text": text, "image": visual, "audio": audio} {
		if len(vector) != dimension {
			t.Fatalf("%s dimension=%d want=%d", modality, len(vector), dimension)
		}
		norm := 0.0
		for _, value := range vector {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				t.Fatalf("nonfinite %s vector", modality)
			}
			norm += float64(value) * float64(value)
		}
		if math.Abs(norm-1) > 1e-3 {
			t.Fatalf("%s squared norm=%g", modality, norm)
		}
	}
	// The source-reference cosine checks representation parity, not recognition accuracy.
	cosine := 0.0
	for i, value := range text {
		cosine += float64(value) * float64(audio[i])
	}
	want := map[string]float64{"nano": .04673499, "mini": .22175031}[variant]
	if math.Abs(cosine-want) > 1e-4 {
		t.Fatalf("text/audio cosine=%g want=%g ± .0001", cosine, want)
	}
	for _, options := range []embedding.Options{{Dimension: dimension / 2}, {Layer: 6}} {
		if _, err := model.EmbedWithOptions(ctx, "hello", options); !errors.Is(err, binding.ErrCapability) {
			t.Fatalf("unsupported representation accepted or wrong error: %v", err)
		}
	}
	if _, err := model.Embed(ctx, strings.Repeat("hello ", limit+16)); !errors.Is(err, binding.ErrInputLimit) {
		t.Fatalf("over-budget text must reject: %v", err)
	}
	if _, err := model.EmbedAudio(ctx, embedding.AudioRequest{PCM: []float32{0}, SampleRate: 44100, Channels: 2}); err == nil {
		t.Fatal("incomplete audio channels accepted")
	}
	if _, err := model.text.Call(ctx, "foreign-recipe", embedding.TextRequest{Text: "hello"}); !errors.Is(err, binding.ErrCapability) {
		t.Fatalf("foreign recipe accepted: %v", err)
	}
	if variant == "mini" {
		result, err := model.text.Call(ctx, variant, embedding.TextRequest{Text: strings.Repeat("hello ", 1024)})
		if err != nil {
			t.Fatal(err)
		}
		if result.Input == nil || result.Input.Truncated || result.Input.ProcessedTokens <= 512 {
			t.Fatalf("longer Mini input did not execute in full: %+v", result.Input)
		}
	}
	return text
}

func publishedOmniTone() embedding.AudioRequest {
	const rate = 44100
	pcm := make([]float32, rate*2)
	for channel, frequency := range []float64{440, 660} {
		for frame := range rate {
			sample := int16(8192 * math.Sin(2*math.Pi*frequency*float64(frame)/rate))
			pcm[channel*rate+frame] = float32(sample) / 32768
		}
	}
	return embedding.AudioRequest{PCM: pcm, SampleRate: rate, Channels: 2}
}
