package catalog

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
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
