package extproc

import (
	"context"
	"encoding/json"
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
	p := &kvPlannerStub{hint: &kvtransfer.Hint{Namespace: "scope", CacheID: "cache", MapperID: "mapper"}}
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
	if h["namespace"] != "scope" || h["cache_id"] != "cache" || h["mapper_id"] != "mapper" || p.calls != 1 || p.dispatch.Principal != "tenant" {
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
