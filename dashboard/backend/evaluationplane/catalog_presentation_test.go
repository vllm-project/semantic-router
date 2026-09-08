package evaluationplane

import (
	"regexp"
	"strings"
	"testing"
)

var catalogCodenamePattern = regexp.MustCompile(`(?i)\b[eg][0-9]\b`)

func TestBuiltinCatalogPresentationUsesProductLanguage(t *testing.T) {
	registry, err := NewRegistry("", "", RegistryOptions{Mixtures: []MixtureTargetSnapshot{
		catalogTestMixtureSnapshot(
			[]ModelArm{catalogTestArm("deep", []string{"text"}), catalogTestArm("fast", []string{"text"})},
			catalogTopologyDigest,
		),
	}})
	if err != nil {
		t.Fatalf("NewRegistry: %v", err)
	}

	catalog := registry.Catalog()
	for _, profile := range catalog.ChangeProfiles {
		assertProductCatalogCopy(t, "change profile "+string(profile.ID), profile.Name, profile.Description)
		for _, slot := range profile.CampaignSlots {
			assertProductCatalogCopy(t, "campaign slot "+slot.GateID, slot.Name, slot.Description)
		}
	}
	for _, track := range catalog.Tracks {
		assertProductCatalogCopy(t, "track "+string(track.ID), track.Name, track.Description)
	}
	for _, suite := range catalog.Suites {
		assertProductCatalogCopy(t, "suite "+suite.ID, suite.Name, suite.Description)
		for _, method := range suite.Methods {
			assertProductCatalogCopy(t, "method reason "+method.ID, method.Reason)
		}
	}
	for _, target := range catalog.Targets {
		assertProductCatalogCopy(t, "target "+target.ID, target.Name, target.Description)
	}
}

func assertProductCatalogCopy(t *testing.T, location string, values ...string) {
	t.Helper()
	for _, value := range values {
		normalized := strings.ToLower(value)
		if catalogCodenamePattern.MatchString(value) {
			t.Fatalf("%s exposes an evidence or release-check codename in %q", location, value)
		}
		for _, internalTerm := range []string{
			"logical arm", "ab/ba", "server-owned", "server-qualified", "ledger",
			"executor", "codename", "vertical slice", "live-mom-core", "brokered", "paired-live",
		} {
			if strings.Contains(normalized, internalTerm) {
				t.Fatalf("%s exposes internal term %q in %q", location, internalTerm, value)
			}
		}
	}
}
