/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package redaction

import (
	"encoding/json"
	"testing"
)

func TestRedactResponseBodyRemovesLooperAttemptDetails(t *testing.T) {
	body := []byte(`{"route_diagnostics":{"selected_model":"large","looper":{"version":1,"trace_id":"trace-1","attempts":[{"ordinal":1,"model":"small"}],"final_attempt_ordinal":1}}}`)

	redacted, changed, err := RedactResponseBody(body)
	if err != nil || !changed {
		t.Fatalf("RedactResponseBody() = changed %v, err %v", changed, err)
	}
	var payload map[string]any
	if err := json.Unmarshal(redacted, &payload); err != nil {
		t.Fatalf("decode redacted response: %v", err)
	}
	diagnostics := payload["route_diagnostics"].(map[string]any)
	looper := diagnostics["looper"].(map[string]any)
	if len(looper) != 0 {
		t.Fatalf("viewer response retained Looper attempt details: %+v", looper)
	}
	if diagnostics["selected_model"] != "large" {
		t.Fatalf("viewer response removed selected model: %+v", diagnostics)
	}
}
