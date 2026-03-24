package classification

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestCategoryKBTaxonomyUnmarshal(t *testing.T) {
	raw := `{
		"categories": {
			"proprietary_code": {"tier": "privacy_policy", "description": "Internal code"},
			"prompt_injection": {"tier": "security_containment", "description": "Injection attacks"},
			"generic_coding":   {"tier": "local_standard", "description": "General coding"}
		},
		"tier_groups": {
			"privacy_categories":  ["proprietary_code"],
			"security_categories": ["prompt_injection"],
			"default_categories":  ["generic_coding"]
		}
	}`

	var taxonomy CategoryKBTaxonomy
	if err := json.Unmarshal([]byte(raw), &taxonomy); err != nil {
		t.Fatalf("Unmarshal taxonomy: %v", err)
	}

	if len(taxonomy.Categories) != 3 {
		t.Errorf("expected 3 categories, got %d", len(taxonomy.Categories))
	}
	if taxonomy.Categories["proprietary_code"].Tier != "privacy_policy" {
		t.Errorf("expected tier privacy_policy, got %q", taxonomy.Categories["proprietary_code"].Tier)
	}
	if len(taxonomy.TierGroups) != 3 {
		t.Errorf("expected 3 tier_groups, got %d", len(taxonomy.TierGroups))
	}
	if len(taxonomy.TierGroups["security_categories"]) != 1 {
		t.Errorf("expected 1 security category, got %d", len(taxonomy.TierGroups["security_categories"]))
	}
}

func TestCategoryKBClassifier_IsCategoryPrivate(t *testing.T) {
	c := &CategoryKBClassifier{
		taxonomy: CategoryKBTaxonomy{
			Categories: map[string]CategoryKBTaxonomyEntry{
				"proprietary_code":  {Tier: "privacy_policy"},
				"prompt_injection":  {Tier: "security_containment"},
				"generic_coding":    {Tier: "local_standard"},
				"architecture_deep": {Tier: "frontier_reasoning"},
			},
		},
		privateTiers: map[string]bool{
			"security_containment": true,
			"privacy_policy":       true,
		},
	}

	tests := []struct {
		category string
		want     bool
	}{
		{"proprietary_code", true},
		{"prompt_injection", true},
		{"generic_coding", false},
		{"architecture_deep", false},
		{"nonexistent", false},
	}

	for _, tt := range tests {
		t.Run(tt.category, func(t *testing.T) {
			got := c.isCategoryPrivate(tt.category)
			if got != tt.want {
				t.Errorf("isCategoryPrivate(%q) = %v, want %v", tt.category, got, tt.want)
			}
		})
	}
}

func TestCategoryKBClassifier_IsCategoryInTierGroup(t *testing.T) {
	c := &CategoryKBClassifier{
		taxonomy: CategoryKBTaxonomy{
			Categories: map[string]CategoryKBTaxonomyEntry{
				"prompt_injection": {Tier: "security_containment"},
				"generic_coding":   {Tier: "local_standard"},
			},
		},
	}

	if !c.isCategoryInTierGroup("prompt_injection", "security_containment") {
		t.Error("expected prompt_injection to be in security_containment")
	}
	if c.isCategoryInTierGroup("generic_coding", "security_containment") {
		t.Error("expected generic_coding NOT in security_containment")
	}
	if c.isCategoryInTierGroup("nonexistent", "security_containment") {
		t.Error("expected nonexistent to not be in any tier group")
	}
}

func TestCategoryKBClassifier_LoadTaxonomy(t *testing.T) {
	dir := t.TempDir()
	taxonomyPath := filepath.Join(dir, "taxonomy.json")
	taxonomyJSON := `{
		"categories": {
			"pii": {"tier": "privacy_policy", "description": "PII data"},
			"jailbreak_role": {"tier": "security_containment"}
		},
		"tier_groups": {
			"privacy_categories": ["pii"],
			"security_categories": ["jailbreak_role"]
		}
	}`

	if err := os.WriteFile(taxonomyPath, []byte(taxonomyJSON), 0o644); err != nil {
		t.Fatal(err)
	}

	c := &CategoryKBClassifier{
		rule: config.CategoryKBRule{
			TaxonomyPath: taxonomyPath,
			KBDir:        dir,
		},
	}

	if err := c.loadTaxonomy(); err != nil {
		t.Fatalf("loadTaxonomy: %v", err)
	}
	if len(c.taxonomy.Categories) != 2 {
		t.Errorf("expected 2 categories, got %d", len(c.taxonomy.Categories))
	}
	if c.taxonomy.Categories["pii"].Tier != "privacy_policy" {
		t.Errorf("pii tier = %q, want privacy_policy", c.taxonomy.Categories["pii"].Tier)
	}
}

func TestCategoryKBClassifier_LoadTaxonomyDefaultPath(t *testing.T) {
	dir := t.TempDir()
	taxonomyJSON := `{"categories": {"test": {"tier": "local_standard"}}}`
	if err := os.WriteFile(filepath.Join(dir, "taxonomy.json"), []byte(taxonomyJSON), 0o644); err != nil {
		t.Fatal(err)
	}

	c := &CategoryKBClassifier{
		rule: config.CategoryKBRule{KBDir: dir},
	}

	if err := c.loadTaxonomy(); err != nil {
		t.Fatalf("loadTaxonomy with default path: %v", err)
	}
	if len(c.taxonomy.Categories) != 1 {
		t.Errorf("expected 1 category, got %d", len(c.taxonomy.Categories))
	}
}

func TestCategoryKBClassifier_LoadKBs(t *testing.T) {
	dir := t.TempDir()

	writeKB := func(t *testing.T, name string, exemplars []string) {
		t.Helper()
		data, err := json.Marshal(map[string]interface{}{
			"category":  name,
			"exemplars": exemplars,
		})
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, name+".json"), data, 0o644); err != nil {
			t.Fatal(err)
		}
	}

	writeKB(t, "proprietary_code", []string{"Review our internal SDK", "Debug our pipeline"})
	writeKB(t, "generic_coding", []string{"How do I sort an array?", "Explain recursion"})

	// taxonomy.json should be skipped
	if err := os.WriteFile(filepath.Join(dir, "taxonomy.json"), []byte(`{"categories":{}}`), 0o644); err != nil {
		t.Fatal(err)
	}

	// empty exemplars should be skipped
	writeKB(t, "empty_cat", []string{})

	c := &CategoryKBClassifier{
		rule: config.CategoryKBRule{KBDir: dir},
		kbs:  make(map[string]*categoryKBData),
	}

	if err := c.loadKBs(); err != nil {
		t.Fatalf("loadKBs: %v", err)
	}

	if len(c.kbs) != 2 {
		t.Errorf("expected 2 KBs loaded, got %d", len(c.kbs))
	}
	if _, ok := c.kbs["proprietary_code"]; !ok {
		t.Error("expected proprietary_code KB to be loaded")
	}
	if _, ok := c.kbs["generic_coding"]; !ok {
		t.Error("expected generic_coding KB to be loaded")
	}
	if _, ok := c.kbs["empty_cat"]; ok {
		t.Error("expected empty_cat to be skipped")
	}
	if c.kbs["proprietary_code"].Exemplars[0] != "Review our internal SDK" {
		t.Errorf("unexpected first exemplar: %q", c.kbs["proprietary_code"].Exemplars[0])
	}
}

func TestCategoryKBClassifier_LoadKBsEmptyDir(t *testing.T) {
	dir := t.TempDir()

	c := &CategoryKBClassifier{
		rule: config.CategoryKBRule{KBDir: dir},
		kbs:  make(map[string]*categoryKBData),
	}

	err := c.loadKBs()
	if err == nil {
		t.Fatal("expected error for empty KB directory")
	}
}

func TestCategoryKBClassifier_CategoryCount(t *testing.T) {
	c := &CategoryKBClassifier{
		kbs: map[string]*categoryKBData{
			"a": {Exemplars: []string{"x"}},
			"b": {Exemplars: []string{"y"}},
		},
	}
	if c.CategoryCount() != 2 {
		t.Errorf("CategoryCount() = %d, want 2", c.CategoryCount())
	}
}

func TestCategoryKBClassifier_ClassifyEmptyQuery(t *testing.T) {
	c := &CategoryKBClassifier{
		kbs: map[string]*categoryKBData{
			"test": {Exemplars: []string{"hello"}},
		},
	}
	_, err := c.Classify("")
	if err == nil {
		t.Error("expected error for empty query")
	}
}

func TestCategoryKBClassifyResult_MatchedRulesThresholding(t *testing.T) {
	c := &CategoryKBClassifier{
		rule: config.CategoryKBRule{
			Threshold:         0.5,
			SecurityThreshold: 0.8,
		},
		kbs: map[string]*categoryKBData{
			"proprietary_code": {
				Exemplars:  []string{"internal code"},
				Embeddings: [][]float32{{0.1, 0.2, 0.3}},
			},
			"prompt_injection": {
				Exemplars:  []string{"ignore previous"},
				Embeddings: [][]float32{{0.4, 0.5, 0.6}},
			},
			"generic_coding": {
				Exemplars:  []string{"sort array"},
				Embeddings: [][]float32{{0.7, 0.8, 0.9}},
			},
		},
		taxonomy: CategoryKBTaxonomy{
			Categories: map[string]CategoryKBTaxonomyEntry{
				"proprietary_code": {Tier: "privacy_policy"},
				"prompt_injection": {Tier: "security_containment"},
				"generic_coding":   {Tier: "local_standard"},
			},
		},
		privateTiers: map[string]bool{
			"security_containment": true,
			"privacy_policy":       true,
		},
	}

	catMaxSim := map[string]float64{
		"proprietary_code": 0.6,
		"prompt_injection": 0.75,
		"generic_coding":   0.9,
	}

	threshold := float64(c.rule.Threshold)
	secThreshold := float64(c.rule.SecurityThreshold)

	matchedRules := make([]string, 0)
	for cat, sim := range catMaxSim {
		applied := threshold
		if c.isCategoryInTierGroup(cat, "security_containment") {
			applied = secThreshold
		}
		if sim >= applied {
			matchedRules = append(matchedRules, cat)
		}
	}

	// proprietary_code: 0.6 >= 0.5 → match
	// prompt_injection: 0.75 < 0.8 → no match (security threshold)
	// generic_coding: 0.9 >= 0.5 → match
	hasProprietaryCode := false
	hasGenericCoding := false
	hasPromptInjection := false
	for _, r := range matchedRules {
		switch r {
		case "proprietary_code":
			hasProprietaryCode = true
		case "generic_coding":
			hasGenericCoding = true
		case "prompt_injection":
			hasPromptInjection = true
		}
	}

	if !hasProprietaryCode {
		t.Error("expected proprietary_code to match (0.6 >= 0.5)")
	}
	if !hasGenericCoding {
		t.Error("expected generic_coding to match (0.9 >= 0.5)")
	}
	if hasPromptInjection {
		t.Error("expected prompt_injection to NOT match (0.75 < 0.8 security threshold)")
	}
}

func TestCategoryKBClassifier_ContrastiveScoring(t *testing.T) {
	c := &CategoryKBClassifier{
		taxonomy: CategoryKBTaxonomy{
			Categories: map[string]CategoryKBTaxonomyEntry{
				"proprietary_code": {Tier: "privacy_policy"},
				"pii":              {Tier: "privacy_policy"},
				"prompt_injection": {Tier: "security_containment"},
				"generic_coding":   {Tier: "local_standard"},
				"architecture":     {Tier: "frontier_reasoning"},
			},
		},
		privateTiers: map[string]bool{
			"security_containment": true,
			"privacy_policy":       true,
		},
	}

	tests := []struct {
		name      string
		catMaxSim map[string]float64
		wantSign  string // "positive", "negative", "zero"
	}{
		{
			name: "privacy dominates",
			catMaxSim: map[string]float64{
				"proprietary_code": 0.9,
				"pii":              0.7,
				"generic_coding":   0.3,
				"architecture":     0.4,
			},
			wantSign: "positive",
		},
		{
			name: "public dominates",
			catMaxSim: map[string]float64{
				"proprietary_code": 0.2,
				"generic_coding":   0.8,
				"architecture":     0.7,
			},
			wantSign: "negative",
		},
		{
			name: "security category also private",
			catMaxSim: map[string]float64{
				"prompt_injection": 0.95,
				"generic_coding":   0.3,
			},
			wantSign: "positive",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var maxPrivate, maxPublic float64
			for cat, sim := range tt.catMaxSim {
				if c.isCategoryPrivate(cat) {
					if sim > maxPrivate {
						maxPrivate = sim
					}
				} else {
					if sim > maxPublic {
						maxPublic = sim
					}
				}
			}
			contrastive := maxPrivate - maxPublic

			switch tt.wantSign {
			case "positive":
				if contrastive <= 0 {
					t.Errorf("expected positive contrastive, got %f", contrastive)
				}
			case "negative":
				if contrastive >= 0 {
					t.Errorf("expected negative contrastive, got %f", contrastive)
				}
			case "zero":
				if contrastive != 0 {
					t.Errorf("expected zero contrastive, got %f", contrastive)
				}
			}
		})
	}
}
