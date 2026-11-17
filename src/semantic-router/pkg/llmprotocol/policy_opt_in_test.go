package llmprotocol

import (
	"strings"
	"testing"
)

func TestOptInHintNamesTheSettingOnlyForAcceptableLosses(t *testing.T) {
	for _, field := range OptInLossyFields() {
		if hint := OptInHint(field); !strings.Contains(hint, field+" in global.router.allow_lossy_translation") {
			t.Errorf("OptInHint(%q) = %q", field, hint)
		}
	}
	if hint := OptInHint("tools.allowed_callers"); hint != "" {
		t.Errorf("a loss no operator may accept got a hint: %q", hint)
	}
}
