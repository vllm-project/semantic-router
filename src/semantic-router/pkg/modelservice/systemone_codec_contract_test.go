package modelservice

import (
	"context"
	"fmt"
	"net/http"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/api"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/systemonecodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/systemoneprotocol"
)

type plannedBatchCodec struct {
	systemonecodec.VLLMSR50Codec
	planned systemoneprotocol.Batch
}

func (c *plannedBatchCodec) Plan(batch systemoneprotocol.Batch, _ systemonecodec.Target, _ systemonecodec.Policy) (systemonecodec.Plan, systemonecodec.Diagnostics, error) {
	c.planned = batch
	return systemonecodec.Plan{}, nil, fmt.Errorf("stop before transport")
}

// A task asked through a schema codec carries its deadline, as the native
// request does, so the runtime can drop queued work it cannot start in time.
func TestCodecBatchCarriesTheTaskDeadline(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), time.Second)
	defer cancel()
	request := Request{State: "short sample", Questions: []Question{{ID: "q", Type: "noul", Instructions: "Is it short?"}}}
	body, err := systemOneBody(ctx, request)
	if err != nil {
		t.Fatal(err)
	}
	call := &bundleCall{
		ctx: ctx, task: api.BundleTask{Decisions: &body},
		done: make(chan struct{}), decision: &decisionCall{request: request},
	}
	codec := &plannedBatchCodec{}
	(&systemOneDeployment{codec: codec}).sendCodec(ctx, fuseDecisions(1, []*bundleCall{call})[0])
	deadline, ok := codec.planned.Options.Deadline()
	if !ok || deadline <= 0 || deadline > 1000 {
		t.Fatalf("batch deadline %v (set %v), want the task's remaining second", deadline, ok)
	}
}

// The runtime answers invalid_model_output with 500 and names the question
// and state, so that failure is the question's own, not the whole call's.
func TestANamedQuestionFailureAt5xxIsTheQuestions(t *testing.T) {
	body := []byte(`{"error":{"code":"question_failed","question":"q","state":"2","question_error":"invalid_model_output","reason":"model_output"}}`)
	failure, ok := failedQuestion(http.StatusInternalServerError, body)
	if !ok || failure.question != "q" || failure.state != "2" || failure.code != "invalid_model_output" {
		t.Fatalf("failedQuestion = %+v, %v", failure, ok)
	}
	if _, ok := failedQuestion(http.StatusInternalServerError, []byte(`{"error":{"code":"internal_error"}}`)); ok {
		t.Fatal("a 500 that names no question was read as one question's failure")
	}
}
