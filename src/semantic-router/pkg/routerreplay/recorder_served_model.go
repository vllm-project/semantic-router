package routerreplay

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"

// ServedModel is the model whose answer a request got and the reasoning it
// was sent with.
type ServedModel = store.ServedModel

// UpdateServedModel names the model that answered a record, such as a
// fallback candidate that answered in place of the routed model.
func (r *Recorder) UpdateServedModel(id string, served ServedModel) error {
	ctx, cancel := r.replayOperationContext()
	defer cancel()
	return r.storage.UpdateServedModel(ctx, id, served)
}
