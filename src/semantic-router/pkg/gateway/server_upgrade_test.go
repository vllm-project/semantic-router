package gateway

import (
	"bufio"
	"fmt"
	"io"
	"net"
	"net/http"
	"strconv"
	"strings"
	"testing"
	"time"

	"golang.org/x/net/http2"
	"golang.org/x/net/http2/hpack"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/upstream"
)

const upgradeTestBody = `{"model":"m"}`

func upgradeTestHandler(t *testing.T, limit int64, timeout time.Duration, keyed bool) *Handler {
	t.Helper()
	keys := []string(nil)
	if keyed {
		keys = []string{"test-key"}
	}
	h, err := NewHandler(Options{
		Serving: Static(Serving{
			Engine:   routing.NewEngine(&fakeProcessor{}, routing.DefaultOptions),
			Upstream: &fakeUpstream{resp: okUpstream("{}", nil)},
			APIKeys:  keys,
		}),
		MaxRequestBodyBytes: limit, ReadTimeout: timeout,
	})
	if err != nil {
		t.Fatal(err)
	}
	return h
}

func upgradeTestHead(framing string) string {
	return "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n" +
		"Connection: Upgrade, HTTP2-Settings\r\nUpgrade: h2c\r\n" +
		"HTTP2-Settings: AAMAAABkAAQAAP__\r\n" + framing + "\r\n"
}

func openUpgradeTest(t *testing.T, addr, head, body string) (net.Conn, *bufio.Reader) {
	t.Helper()
	conn, err := net.DialTimeout("tcp", addr, time.Second)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = conn.Close() })
	_ = conn.SetDeadline(time.Now().Add(time.Second))
	if _, err := io.WriteString(conn, upgradeTestHead(head)+body); err != nil {
		t.Fatal(err)
	}
	return conn, bufio.NewReader(conn)
}

func assertUpgradeRefused(t *testing.T, reader *bufio.Reader, status int, code string) {
	t.Helper()
	response, err := http.ReadResponse(reader, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	if response.StatusCode != status {
		t.Fatalf("initial response = %d, want %d before Upgrade", response.StatusCode, status)
	}
	if !response.Close {
		t.Fatal("an unread Upgrade body must close the connection")
	}
	body, err := io.ReadAll(response.Body)
	if err != nil || !strings.Contains(string(body), `"code":"`+code+`"`) {
		t.Fatalf("refusal body = %q, err = %v, want code %s", body, err, code)
	}
}

func finishUpgradeTest(t *testing.T, conn net.Conn, reader *bufio.Reader) (int, string) {
	t.Helper()
	response, err := http.ReadResponse(reader, nil)
	if err != nil || response.StatusCode != http.StatusSwitchingProtocols {
		t.Fatalf("Upgrade response = %v, err = %v", response, err)
	}
	defer response.Body.Close()
	if _, writeErr := io.WriteString(conn, http2.ClientPreface); writeErr != nil {
		t.Fatal(writeErr)
	}
	frames := http2.NewFramer(conn, reader)
	frames.ReadMetaHeaders = hpack.NewDecoder(4096, nil)
	if settingsErr := frames.WriteSettings(); settingsErr != nil {
		t.Fatal(settingsErr)
	}
	status := 0
	var body strings.Builder
	for {
		frame, frameErr := frames.ReadFrame()
		if frameErr != nil {
			t.Fatal(frameErr)
		}
		switch f := frame.(type) {
		case *http2.SettingsFrame:
			if !f.IsAck() {
				if ackErr := frames.WriteSettingsAck(); ackErr != nil {
					t.Fatal(ackErr)
				}
			}
		case *http2.MetaHeadersFrame:
			if f.StreamID == 1 {
				if value := f.PseudoValue("status"); value != "" {
					status, err = strconv.Atoi(value)
					if err != nil {
						t.Fatal(err)
					}
				}
				if f.StreamEnded() {
					return status, body.String()
				}
			}
		case *http2.DataFrame:
			if f.StreamID == 1 {
				body.Write(f.Data())
				if f.StreamEnded() {
					return status, body.String()
				}
			}
		case *http2.RSTStreamFrame:
			t.Fatalf("response stream reset: %v", f.ErrCode)
		}
	}
}

func TestH2CUpgradeRejectsOversizedBodiesBeforeBuffering(t *testing.T) {
	for _, chunked := range []bool{false, true} {
		for _, keyed := range []bool{false, true} {
			t.Run(fmt.Sprintf("chunked=%t/keyed=%t", chunked, keyed), func(t *testing.T) {
				h := upgradeTestHandler(t, 64, 50*time.Millisecond, keyed)
				addr, stop := serveForTest(t, h, ServerOptions{})
				defer stop()
				// Do not send a whole body: neither a large declaration nor a
				// chunk exceeding the cap may wait for its remaining bytes.
				head, body := "Content-Length: 32768\r\n", "{"
				if chunked {
					head, body = "Transfer-Encoding: chunked\r\n", "8000\r\n"+strings.Repeat("x", 65)
				}
				_, reader := openUpgradeTest(t, addr, head, body)
				assertUpgradeRefused(t, reader, http.StatusRequestEntityTooLarge, "request_too_large")
			})
		}
	}
}

func TestH2CUpgradeTimesOutBeforeGatewayProcessing(t *testing.T) {
	for _, chunked := range []bool{false, true} {
		for _, keyed := range []bool{false, true} {
			t.Run(fmt.Sprintf("chunked=%t/keyed=%t", chunked, keyed), func(t *testing.T) {
				addr, stop := serveForTest(t, upgradeTestHandler(t, 64, 50*time.Millisecond, keyed), ServerOptions{})
				defer stop()
				head, body := "Content-Length: 32\r\n", "{"
				if chunked {
					head, body = "Transfer-Encoding: chunked\r\n", "20\r\n{"
				}
				_, reader := openUpgradeTest(t, addr, head, body)
				assertUpgradeRefused(t, reader, http.StatusRequestTimeout, "request_timeout")
			})
		}
	}
}

func TestH2CUpgradeRejectsMalformedBody(t *testing.T) {
	addr, stop := serveForTest(t, upgradeTestHandler(t, 64, time.Second, false), ServerOptions{})
	defer stop()
	_, reader := openUpgradeTest(t, addr, "Transfer-Encoding: chunked\r\n", "not-hex\r\n")
	assertUpgradeRefused(t, reader, http.StatusBadRequest, "invalid_request_body")
}

func TestH2CUpgradePreservesValidRequestsAtLimit(t *testing.T) {
	for _, chunked := range []bool{false, true} {
		for _, auth := range []string{"disabled", "missing", "valid"} {
			t.Run(fmt.Sprintf("chunked=%t/auth=%s", chunked, auth), func(t *testing.T) {
				keyed := auth != "disabled"
				addr, stop := serveForTest(t, upgradeTestHandler(t, int64(len(upgradeTestBody)), time.Second, keyed), ServerOptions{})
				defer stop()
				head, body := fmt.Sprintf("Content-Length: %d\r\n", len(upgradeTestBody)), upgradeTestBody
				if chunked {
					head, body = "Transfer-Encoding: chunked\r\n", fmt.Sprintf("%x\r\n%s\r\n0\r\n\r\n", len(body), body)
				}
				if auth == "valid" {
					head += "Authorization: Bearer test-key\r\n"
				}
				conn, reader := openUpgradeTest(t, addr, head, body)
				status, result := finishUpgradeTest(t, conn, reader)
				want := http.StatusOK
				if auth == "missing" {
					want = http.StatusUnauthorized
				}
				if status != want || (want == http.StatusOK && result != "{}") {
					t.Fatalf("response = %d %q, want %d", status, result, want)
				}
			})
		}
	}
}

func TestH2CUpgradeRefusalClosesWithTimeoutDisabled(t *testing.T) {
	addr, stop := serveForTest(t, upgradeTestHandler(t, 64, 0, false), ServerOptions{})
	defer stop()
	_, reader := openUpgradeTest(t, addr, "Content-Length: 65\r\n", "{")
	assertUpgradeRefused(t, reader, http.StatusRequestEntityTooLarge, "request_too_large")
	if _, readErr := reader.ReadByte(); readErr != io.EOF {
		t.Fatalf("connection after refusal = %v, want EOF", readErr)
	}
}

func TestH2CUpgradeStreamsPastReadTimeout(t *testing.T) {
	processor := &fakeProcessor{effects: map[routing.Phase][]*routing.Effect{
		routing.PhaseResponseHeaders: {{ResponseBodyMode: routing.BodyModeStreamed}},
	}}
	reader, writer := io.Pipe()
	defer reader.Close()
	defer writer.Close()
	go func() {
		_, _ = io.WriteString(writer, "data: one\n\n")
		time.Sleep(100 * time.Millisecond)
		_, _ = io.WriteString(writer, "data: [DONE]\n\n")
		_ = writer.Close()
	}()
	h := upgradeTestHandler(t, 64, 40*time.Millisecond, false)
	h.opts.Serving = Static(Serving{
		Engine: routing.NewEngine(processor, routing.DefaultOptions),
		Upstream: &fakeUpstream{resp: func() *upstream.Response {
			return &upstream.Response{StatusCode: http.StatusOK, Header: http.Header{"Content-Type": {"text/event-stream"}}, Body: reader}
		}},
	})
	addr, stop := serveForTest(t, h, ServerOptions{})
	defer stop()
	conn, response := openUpgradeTest(t, addr, fmt.Sprintf("Content-Length: %d\r\n", len(upgradeTestBody)), upgradeTestBody)
	if status, body := finishUpgradeTest(t, conn, response); status != http.StatusOK || body != "data: one\n\ndata: [DONE]\n\n" {
		t.Fatalf("streamed response = %d %q", status, body)
	}
}

func TestH2CUpgradePreservesAdapterErrors(t *testing.T) {
	addr, stop := serveForTest(t, upgradeTestHandler(t, 64, 0, false), ServerOptions{})
	defer stop()
	_, reader := openUpgradeTest(t, addr, "HTTP2-Settings: invalid\r\nContent-Length: 32\r\n", "{")
	response, err := http.ReadResponse(reader, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusInternalServerError || !response.Close {
		t.Fatalf("adapter refusal = %d, close=%t", response.StatusCode, response.Close)
	}
	if _, readErr := io.ReadAll(response.Body); readErr != nil {
		t.Fatal(readErr)
	}
}

func TestH2CUpgradeReadDeadlineEndsWithBody(t *testing.T) {
	h := upgradeTestHandler(t, 64, 40*time.Millisecond, false)
	h.opts.Serving = Static(Serving{Engine: routing.NewEngine(&fakeProcessor{}, routing.DefaultOptions), Upstream: slowUpstream{}})
	addr, stop := serveForTest(t, h, ServerOptions{})
	defer stop()
	conn, reader := openUpgradeTest(t, addr, fmt.Sprintf("Content-Length: %d\r\n", len(upgradeTestBody)), upgradeTestBody)
	if status, body := finishUpgradeTest(t, conn, reader); status != http.StatusOK || body != "{}" {
		t.Fatalf("slow response = %d %q", status, body)
	}
}

func TestH2CUpgradePreservesDisabledReadTimeout(t *testing.T) {
	addr, stop := serveForTest(t, upgradeTestHandler(t, 64, 0, false), ServerOptions{IdleTimeout: 40 * time.Millisecond})
	defer stop()
	conn, reader := openUpgradeTest(t, addr, fmt.Sprintf("Content-Length: %d\r\n", len(upgradeTestBody)), upgradeTestBody[:1])
	time.Sleep(100 * time.Millisecond)
	if _, err := io.WriteString(conn, upgradeTestBody[1:]); err != nil {
		t.Fatal(err)
	}
	if status, body := finishUpgradeTest(t, conn, reader); status != http.StatusOK || body != "{}" {
		t.Fatalf("response with timeout disabled = %d %q", status, body)
	}
}
