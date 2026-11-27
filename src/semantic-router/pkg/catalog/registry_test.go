package catalog

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestBuiltInDigestMatchesPublishedCatalog(t *testing.T) {
	payload, err := os.ReadFile(filepath.Join("..", "..", "..", "..", "website", "static", "model-catalog", "catalog.json"))
	if err != nil {
		t.Fatalf("read published catalog: %v", err)
	}
	sum := sha256.Sum256(payload)
	want := "sha256:" + hex.EncodeToString(sum[:])

	registry, err := BuiltIn()
	if err != nil {
		t.Fatal(err)
	}
	if got := registry.Digest(); got != want {
		t.Fatalf("built-in digest = %s, want %s", got, want)
	}
	effective, err := registry.Compile(CompileInput{})
	if err != nil {
		t.Fatal(err)
	}
	if got := effective.Digest(); got != want {
		t.Fatalf("effective digest = %s, want %s", got, want)
	}
}

func TestBuiltInGemma4ITVLLMBindings(t *testing.T) {
	registry, err := BuiltIn()
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		checkpoint string
		context    int
	}{
		{"google/gemma-4-12B-it", 262144},
		{"google/gemma-4-26B-A4B-it", 262144},
		{"google/gemma-4-E4B-it", 131072},
		{"google/gemma-4-E2B-it", 131072},
	} {
		t.Run(tc.checkpoint, func(t *testing.T) {
			effective, err := registry.Compile(CompileInput{
				Providers: []ProviderInstance{{Name: "local", Catalog: "vllm", BaseURL: "http://127.0.0.1:8000"}},
				Models: []ModelAlias{{
					Name: "gemma", Catalog: strings.ToLower(tc.checkpoint),
					Providers: []ModelProviderBinding{{Name: "local"}},
				}},
			})
			if err != nil {
				t.Fatal(err)
			}
			model, ok := effective.Model("gemma")
			if !ok || len(model.Providers) != 1 {
				t.Fatalf("compiled provider binding is missing: %+v", model)
			}
			card := model.Card.Card
			if card.Distribution.License != "Apache-2.0" || card.Limits.ContextWindowSize != tc.context || card.ReasoningFamily != "gemma4" {
				t.Fatalf("unexpected model card: %+v", card)
			}
			binding := model.Providers[0]
			if binding.Binding.ModelID != tc.checkpoint || binding.Binding.Protocol != "openai/chat-completions@1" {
				t.Fatalf("binding = %+v, want Chat Completions for %q", binding.Binding, tc.checkpoint)
			}
			if binding.CatalogBinding == nil || binding.CatalogBinding.ReasoningTransport != "chat_template_kwargs" {
				t.Fatalf("missing vLLM thinking transport: %+v", binding.CatalogBinding)
			}
		})
	}
}

// A model reached with a Claude or ChatGPT login has no operator pricing, so
// its cost and savings come only from its first-party binding's list price.
func TestBuiltInAnthropicAndOpenAIBindingsCarryListPrices(t *testing.T) {
	registry, err := BuiltIn()
	if err != nil {
		t.Fatal(err)
	}
	for _, providerID := range []string{"anthropic", "openai"} {
		provider, ok := registry.Provider(providerID)
		if !ok {
			t.Fatalf("provider %q is missing", providerID)
		}
		for _, binding := range provider.Models {
			if binding.Relationship != CatalogModelRelationshipFirstParty || binding.Lifecycle != "active" {
				continue
			}
			price := binding.Pricing
			if price.Currency != "USD" || price.PromptPer1M <= 0 || price.CompletionPer1M <= 0 || price.CachedInputPer1M <= 0 {
				t.Errorf("%s %s pricing = %+v, want USD input, output and cache-read rates", providerID, binding.ID, price)
			}
			if providerID == "anthropic" && (price.CacheWritePer1M == nil || price.CacheWrite1hPer1M == nil) {
				t.Errorf("%s pricing has no 5-minute and 1-hour cache-write rates", binding.ID)
			}
		}
		for _, entry := range provider.PriceList {
			if price := entry.Pricing; price.Currency != "USD" || price.PromptPer1M <= 0 || price.CompletionPer1M <= 0 || price.CachedInputPer1M <= 0 {
				t.Errorf("%s price list %s = %+v, want USD input, output and cache-read rates", providerID, entry.ID, price)
			}
		}
	}
	openai, _ := registry.Provider("openai")
	if listed := len(openai.PriceList); listed == 0 {
		t.Fatal("openai lists no price for a model without a card")
	}
}
