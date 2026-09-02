package config

import "testing"

// The schema tests parse config/config.yaml, they do not run it through Load,
// so a signal that is in the canonical schema but never copied into Signals
// looks fine to them and arrives empty at runtime. This loads it the way the
// router does.
func TestReferenceConfigLoadsAndValidates(t *testing.T) {
	root := repoRootFromTestFile(t)
	cfg, err := Load(root + "/config/config.yaml")
	if err != nil {
		t.Fatalf("reference config failed to load: %v", err)
	}
	if len(cfg.Signals.ResponseJailbreakRules) == 0 {
		t.Error("response_jailbreak rules declared in the reference config did not survive the load")
	}
}
