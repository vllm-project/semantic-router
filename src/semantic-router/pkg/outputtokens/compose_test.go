package outputtokens

import "testing"

func TestComposeSelectsStrictestBound(t *testing.T) {
	client := int64(9000)
	plugin := int64(8192)
	model := int64(256)
	stage := int64(512)
	result := Compose(Sources{
		Client:         &client,
		Plugin:         &plugin,
		ModelRef:       &model,
		AlgorithmStage: &stage,
	})
	if result.Effective == nil || *result.Effective != 256 || result.Source != SourceModelRef {
		t.Fatalf("compose = %+v, want 256 from model_ref", result)
	}
}

func TestComposeClientCannotWiden(t *testing.T) {
	client := int64(100)
	model := int64(512)
	result := Compose(Sources{Client: &client, ModelRef: &model})
	if result.Effective == nil || *result.Effective != 100 || result.Source != SourceClient {
		t.Fatalf("compose = %+v, want client 100", result)
	}
}

func TestComposeOmitsWhenNoSources(t *testing.T) {
	result := Compose(Sources{})
	if result.Effective != nil || result.Source != "" {
		t.Fatalf("compose = %+v, want empty", result)
	}
}

func TestComposeIgnoresNonPositive(t *testing.T) {
	zero := int64(0)
	negative := int64(-8)
	model := int64(128)
	result := Compose(Sources{Client: &zero, Plugin: &negative, ModelRef: &model})
	if result.Effective == nil || *result.Effective != 128 || result.Source != SourceModelRef {
		t.Fatalf("compose = %+v, want model_ref 128", result)
	}
}

func TestComposeInjectsPluginWhenClientOmitted(t *testing.T) {
	plugin := int64(8192)
	result := Compose(Sources{Plugin: &plugin})
	if result.Effective == nil || *result.Effective != 8192 || result.Source != SourcePlugin {
		t.Fatalf("compose = %+v, want plugin 8192", result)
	}
}

type testLedger struct {
	value int64
	ok    bool
}

func (ledger testLedger) UpperBound() (int64, string, bool) {
	return ledger.value, SourceLedger, ledger.ok
}

func TestComposeIncludesLedgerContributor(t *testing.T) {
	client := int64(400)
	result := Compose(Sources{Client: &client, Ledger: testLedger{value: 64, ok: true}})
	if result.Effective == nil || *result.Effective != 64 || result.Source != SourceLedger {
		t.Fatalf("compose = %+v, want ledger 64", result)
	}
}

func TestComposeSkipsUnavailableLedger(t *testing.T) {
	client := int64(400)
	result := Compose(Sources{Client: &client, Ledger: testLedger{value: 64, ok: false}})
	if result.Effective == nil || *result.Effective != 400 || result.Source != SourceClient {
		t.Fatalf("compose = %+v, want client 400", result)
	}
}
