//go:build !windows && cgo && (amd64 || arm64)

package instance

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

type publishedOmniText struct {
	Text          string    `json:"text"`
	FormattedText string    `json:"formatted_text"`
	Tokens        int       `json:"tokens"`
	Embedding     omniArray `json:"embedding"`
}

type publishedOmniGoldens struct {
	Texts  []publishedOmniText `json:"texts"`
	Images []struct {
		File      string    `json:"file"`
		Embedding omniArray `json:"embedding"`
	} `json:"images"`
	Audio []struct {
		SamplingRate int       `json:"sampling_rate"`
		PCM          omniArray `json:"pcm"`
		Embedding    omniArray `json:"embedding"`
	} `json:"audio"`
}

func readPublishedOmniGoldens(t *testing.T, root string) (string, publishedOmniGoldens) {
	t.Helper()
	golden := filepath.Join(root, "golden")
	raw, err := os.ReadFile(filepath.Join(golden, "index.json"))
	if err != nil {
		t.Fatal(err)
	}
	var index publishedOmniGoldens
	if err := json.Unmarshal(raw, &index); err != nil {
		t.Fatal(err)
	}
	if len(index.Texts) < 3 || len(index.Images) < 2 || len(index.Audio) < 4 {
		t.Fatal("reference lacks public modality coverage")
	}
	return golden, index
}

// Qualify the actual maximum independently of a deployment's smaller input
// budget. Only this maximum-length golden runs; short inputs are not repeatedly
// padded to the full GPU context merely to duplicate the modality parity test.
func TestPublishedOmniFullContext(t *testing.T) {
	options := publishedOmniOptions(t)
	golden, longest := fullContextReference(t, options.ModelPath)
	model, err := LoadOmni(options)
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	info, err := model.Info()
	if err != nil {
		t.Fatal(err)
	}
	if longest.Tokens != info.ModelLimit || info.EffectiveLimit != info.ModelLimit {
		t.Fatalf("full-context qualification requires a %d-token golden and deployment: golden=%d, effective=%d", info.ModelLimit, longest.Tokens, info.EffectiveLimit)
	}
	if options.Provider == "cpu" && options.ExecutionMaxInputTokens != 0 {
		t.Fatal("CPU full-context qualification must retain dynamic input shapes")
	}
	if options.Provider != "cpu" && options.ExecutionMaxInputTokens != info.ModelLimit {
		t.Fatal("GPU full-context qualification requires the explicit full execution budget")
	}
	out, err := model.EncodeText(longest.Text, 0)
	if err != nil {
		t.Fatal(err)
	}
	if out.Input == nil || out.Input.Truncated || out.Input.OriginalTokens != info.ModelLimit || out.Input.ProcessedTokens != info.ModelLimit {
		t.Fatalf("full-context input was not executed completely: %+v", out.Input)
	}
	compareOmniVector(t, "full-context", out.Values, readOmniArray(t, golden, longest.Embedding))
	info, err = model.Info()
	if err != nil || info.CompletedInferences != 1 {
		t.Fatalf("missing full-context execution evidence: %+v %v", info, err)
	}
	t.Logf("qualified %d tokens on %s, execution budget %d", info.ModelLimit, options.Provider, options.ExecutionMaxInputTokens)
	auditPublishedOmniProfiles(t, model, options, info)
}

// A separately generated source reference can qualify the same immutable
// artifact without rewriting its export receipt. The pinned source identity
// must match; this path cannot turn an unsealed export into a loadable model.
func fullContextReference(t *testing.T, artifact string) (string, publishedOmniText) {
	t.Helper()
	path := os.Getenv("VELA_OMNI_FULL_CONTEXT_REFERENCE")
	if path == "" {
		golden, index := readPublishedOmniGoldens(t, artifact)
		var longest publishedOmniText
		for _, record := range index.Texts {
			if record.Tokens > longest.Tokens {
				longest = record
			}
		}
		return golden, longest
	}
	type sourceIdentity struct {
		RepoID   string `json:"repo_id"`
		Revision string `json:"revision"`
	}
	var reference struct {
		Schema string
		Source sourceIdentity
		publishedOmniText
	}
	var manifest struct{ Source sourceIdentity }
	for file, target := range map[string]any{path: &reference, filepath.Join(artifact, "vela_omni_manifest.json"): &manifest} {
		data, err := os.ReadFile(file)
		if err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(data, target); err != nil {
			t.Fatal(err)
		}
	}
	if reference.Schema != "vela_omni_full_context_reference.v1" || reference.Source != manifest.Source || reference.Source.Revision == "" || reference.Text == "" {
		t.Fatal("full-context reference does not describe the artifact's pinned source")
	}
	return filepath.Dir(path), reference.publishedOmniText
}
