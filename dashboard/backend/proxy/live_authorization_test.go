package proxy

import (
	"bufio"
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"net/url"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestLiveAuthorizationRejectsBeforeProxying(t *testing.T) {
	var called atomic.Bool
	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	response := httptest.NewRecorder()
	ServeWithLiveAuthorization(response, request, http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		called.Store(true)
	}), func(context.Context) error { return errors.New("session revoked") }, 10*time.Millisecond)
	if response.Code != http.StatusForbidden || called.Load() {
		t.Fatalf("status=%d upstream=%v, want 403 and no upstream call", response.Code, called.Load())
	}
}

func TestLiveAuthorizationDropsEventsAfterRevocation(t *testing.T) {
	releaseSecond := make(chan struct{})
	upstreamCanceled := make(chan struct{})
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer close(upstreamCanceled)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: first\n\n")
		w.(http.Flusher).Flush()
		select {
		case <-releaseSecond:
			_, _ = io.WriteString(w, "data: second\n\n")
			w.(http.Flusher).Flush()
		case <-r.Context().Done():
			return
		}
		<-r.Context().Done()
	}))
	defer upstream.Close()

	var revoked atomic.Bool
	front := newLiveAuthorizedProxy(t, upstream.URL, &revoked)
	defer front.Close()
	client := front.Client()
	client.Timeout = 10 * time.Second
	response, err := client.Get(front.URL + "/v1/chat/completions")
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		t.Fatalf("status=%d", response.StatusCode)
	}
	reader := bufio.NewReader(response.Body)
	if line, readErr := reader.ReadString('\n'); readErr != nil || line != "data: first\n" {
		t.Fatalf("first event=%q error=%v", line, readErr)
	}
	if line, readErr := reader.ReadString('\n'); readErr != nil || line != "\n" {
		t.Fatalf("first event terminator=%q error=%v", line, readErr)
	}

	revoked.Store(true)
	close(releaseSecond)
	remaining := make(chan string, 1)
	go func() {
		bytes, _ := io.ReadAll(reader)
		remaining <- string(bytes)
	}()
	select {
	case tail := <-remaining:
		if strings.Contains(tail, "second") {
			t.Fatalf("post-revocation event escaped: %q", tail)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("revoked response did not close")
	}
	select {
	case <-upstreamCanceled:
	case <-time.After(2 * time.Second):
		t.Fatal("revoked upstream request did not close")
	}
}

func TestLiveAuthorizationCancelsIdleUpstream(t *testing.T) {
	upstreamCanceled := make(chan struct{})
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: first\n\n")
		w.(http.Flusher).Flush()
		<-r.Context().Done()
		close(upstreamCanceled)
	}))
	defer upstream.Close()

	var revoked atomic.Bool
	front := newLiveAuthorizedProxy(t, upstream.URL, &revoked)
	defer front.Close()
	client := front.Client()
	client.Timeout = 10 * time.Second
	response, err := client.Get(front.URL + "/v1/chat/completions")
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	reader := bufio.NewReader(response.Body)
	if line, readErr := reader.ReadString('\n'); readErr != nil || line != "data: first\n" {
		t.Fatalf("first event=%q error=%v", line, readErr)
	}

	revoked.Store(true)
	select {
	case <-upstreamCanceled:
	case <-time.After(2 * time.Second):
		t.Fatal("idle upstream survived permission revocation")
	}
}

func newLiveAuthorizedProxy(t *testing.T, upstreamURL string, revoked *atomic.Bool) *httptest.Server {
	t.Helper()
	target, err := url.Parse(upstreamURL)
	if err != nil {
		t.Fatal(err)
	}
	reverseProxy := httputil.NewSingleHostReverseProxy(target)
	reverseProxy.FlushInterval = -1
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ServeWithLiveAuthorization(w, r, reverseProxy, func(context.Context) error {
			if revoked.Load() {
				return errors.New("permission revoked")
			}
			return nil
		}, 10*time.Millisecond)
	}))
}
