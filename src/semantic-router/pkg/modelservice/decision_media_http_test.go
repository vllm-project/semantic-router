package modelservice

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/api"
)

func TestDecisionMediaHTTPTransport(t *testing.T) {
	runtime := fakeModels([]modelEntry{{Name: "kai", Model: "vllm-sr/Decision-2.0-Kai-0.6B"}}).Handler()
	var mu sync.Mutex
	var seen []api.DecisionRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		data, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, "read", 500)
			return
		}
		r.Body.Close()
		r.Body = io.NopCloser(bytes.NewReader(data))
		if r.URL.Path == "/v1/decisions" {
			var body api.DecisionRequest
			if err := json.Unmarshal(data, &body); err != nil {
				http.Error(w, "json", 400)
				return
			}
			mu.Lock()
			seen = append(seen, body)
			mu.Unlock()
		}
		runtime.ServeHTTP(w, r)
	}))
	defer server.Close()
	client, err := NewClient(server.URL)
	if err != nil {
		t.Fatal(err)
	}
	for _, enabled := range []bool{false, true} {
		request := sampleRequest("same text")
		request.Model = "kai"
		if enabled {
			request.Images = []string{"data:image/png;base64,iVBORw0KGgo="}
			request.Videos = []string{"data:video/mp4;base64,AAAAFGZ0eXBpc29t"}
		}
		if _, err := client.Decide(context.Background(), request); err != nil {
			t.Fatal(err)
		}
	}
	mu.Lock()
	defer mu.Unlock()
	if len(seen) != 2 || seen[0].Images != nil || seen[0].Videos != nil {
		t.Fatal("text-only request changed")
	}
	if seen[1].Images == nil || len(*seen[1].Images) != 1 || seen[1].Videos == nil || len(*seen[1].Videos) != 1 {
		t.Fatal("media lost on HTTP boundary")
	}
}

func TestDecisionMediaSerializedBodyLimit(t *testing.T) {
	client := &Client{}
	request := sampleRequest("state")
	request.Images = []string{"data:image/png;base64," + strings.Repeat("A", 128)}
	wire, err := encodeDecisionRequest(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	body, _ := json.Marshal(wire)
	client.requestBytes.Store(int64(len(body)))
	if err := client.checkMediaBody(wire, true); err != nil {
		t.Fatal("exact boundary rejected", err)
	}
	client.requestBytes.Store(int64(len(body) - 1))
	if err := client.checkMediaBody(wire, true); !errors.Is(err, ErrRejected) {
		t.Fatal("oversized body accepted", err)
	}
	if err := client.checkMediaBody(wire, false); err != nil {
		t.Fatal("default-off behavior changed")
	}
}
