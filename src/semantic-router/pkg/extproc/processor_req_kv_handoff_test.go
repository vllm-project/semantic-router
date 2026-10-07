package extproc

import (
	"context"
	"encoding/json"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/kvtransfer"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"testing"
)

type kvPlannerStub struct {
	hint     *kvtransfer.Hint
	calls    int
	dispatch KVDispatch
}

func (p *kvPlannerStub) PlanDispatch(_ context.Context, d KVDispatch) (*kvtransfer.Hint, kvtransfer.Reason) {
	p.calls++
	p.dispatch = d
	return p.hint, kvtransfer.ReasonEligible
}
func TestKVHandoffSelectedSwitch(t *testing.T) {
	t.Setenv("USER_SCOPE_NAMESPACE_SECRET", "test-only-secret")
	p := &kvPlannerStub{hint: &kvtransfer.Hint{Namespace: cache.UserScopeNamespace("tenant"), CacheID: "cache", MapperID: "mapper"}}
	r := &OpenAIRouter{KVHandoff: p}
	c := &RequestContext{AuthenticatedPrincipal: "tenant", SessionID: "session", SessionProvenance: SessionProvenanceHeader, PreviousModel: "source", RequestModel: "target", TurnIndex: 1}
	b, e := r.encodeKVHandoff([]byte(`{"model":"target","messages":[]}`), llmprotocol.OpenAIChatV1, c)
	if e != nil {
		t.Fatal(e)
	}
	var w map[string]json.RawMessage
	if e = json.Unmarshal(b, &w); e != nil {
		t.Fatal(e)
	}
	var h map[string]string
	if e = json.Unmarshal(w["kv_transfer_params"], &h); e != nil {
		t.Fatal(e)
	}
	if h["namespace"] != cache.UserScopeNamespace("tenant") || h["cache_id"] != "cache" || h["mapper_id"] != "mapper" || p.calls != 1 || p.dispatch.Principal != "tenant" {
		t.Fatalf("hint %v dispatch %+v", h, p.dispatch)
	}
}
func TestKVHandoffStripsCallerHint(t *testing.T) {
	r := &OpenAIRouter{KVHandoff: &kvPlannerStub{}}
	b, e := r.encodeKVHandoff([]byte(`{"model":"target","kv_transfer_params":{"namespace":"other"}}`), llmprotocol.OpenAIChatV1, &RequestContext{RequestModel: "target", PreviousModel: "target"})
	if e != nil {
		t.Fatal(e)
	}
	var w map[string]json.RawMessage
	if e = json.Unmarshal(b, &w); e != nil {
		t.Fatal(e)
	}
	if _, ok := w["kv_transfer_params"]; ok {
		t.Fatal("caller hint survived")
	}
}
func TestKVHandoffSkipPreservesRequest(t *testing.T) {
	p := &kvPlannerStub{}
	r := &OpenAIRouter{KVHandoff: p}
	original := []byte(`{"model":"target", "messages":[]}`)
	b, e := r.encodeKVHandoff(original, llmprotocol.OpenAIChatV1, &RequestContext{RequestModel: "target", PreviousModel: "target"})
	if e != nil || string(b) != string(original) || p.calls != 0 {
		t.Fatalf("skip changed request %s %v", b, e)
	}
}

func TestKVHandoffRejectsWrongScope(t *testing.T) {
	t.Setenv("USER_SCOPE_NAMESPACE_SECRET", "test-only-secret")
	p := &kvPlannerStub{hint: &kvtransfer.Hint{Namespace: "another-tenant", CacheID: "cache", MapperID: "mapper"}}
	r := &OpenAIRouter{KVHandoff: p}
	c := &RequestContext{AuthenticatedPrincipal: "tenant", SessionID: "session", SessionProvenance: SessionProvenanceHeader, PreviousModel: "source", RequestModel: "target"}
	original := []byte(`{"model":"target"}`)
	b, err := r.encodeKVHandoff(original, llmprotocol.OpenAIChatV1, c)
	if err != nil || string(b) != string(original) {
		t.Fatalf("wrong-scope hint was forwarded: %s %v", b, err)
	}
}

func TestKVHandoffLooperStripsCallerHint(t *testing.T) {
	p := &kvPlannerStub{}
	r := &OpenAIRouter{KVHandoff: p}
	c := &RequestContext{LooperRequest: true, PreviousModel: "source", RequestModel: "target"}
	b, err := r.encodeKVHandoff([]byte(`{"model":"target","kv_transfer_params":{"namespace":"attacker"}}`), llmprotocol.OpenAIChatV1, c)
	if err != nil {
		t.Fatal(err)
	}
	var w map[string]json.RawMessage
	if err = json.Unmarshal(b, &w); err != nil {
		t.Fatal(err)
	}
	if _, ok := w["kv_transfer_params"]; ok || p.calls != 0 {
		t.Fatal("looper forwarded caller hint or planned handoff")
	}
}
