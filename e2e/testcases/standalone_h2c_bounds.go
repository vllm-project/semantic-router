package testcases

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"time"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("standalone-h2c-body-limit", pkgtestcases.TestCase{
		Description: "An oversized h2c Upgrade declaration is refused before reading the body or switching protocols",
		Tags:        []string{"standalone", "gateway", "h2c", "kubernetes"},
		Priority:    pkgtestcases.P0,
		Fn:          testStandaloneH2CBodyLimit,
	})
}

func testStandaloneH2CBodyLimit(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()
	probeCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	conn, err := (&net.Dialer{}).DialContext(probeCtx, "tcp", net.JoinHostPort("localhost", session.LocalPort()))
	if err != nil {
		return err
	}
	defer conn.Close()
	stop := context.AfterFunc(probeCtx, func() { _ = conn.Close() })
	defer stop()
	deadline, _ := probeCtx.Deadline()
	if deadlineErr := conn.SetDeadline(deadline); deadlineErr != nil {
		return deadlineErr
	}
	// The native gateway's default cap is 500 MiB. Declare one byte more,
	// but send only one byte: the server must refuse without buffering it.
	request := "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n" +
		"Connection: Upgrade, HTTP2-Settings\r\nUpgrade: h2c\r\n" +
		"HTTP2-Settings: AAMAAABkAAQAAP__\r\nContent-Length: 524288001\r\n\r\n{"
	if _, writeErr := io.WriteString(conn, request); writeErr != nil {
		return writeErr
	}
	response, err := http.ReadResponse(bufio.NewReader(conn), nil)
	if err != nil {
		return fmt.Errorf("h2c upgrade was not refused within the five-second probe budget: %w", err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusRequestEntityTooLarge || !response.Close {
		return fmt.Errorf("h2c upgrade response = %d, close=%t; want 413 and connection close before 101", response.StatusCode, response.Close)
	}
	var payload struct {
		Error struct {
			Code string `json:"code"`
		} `json:"error"`
	}
	if decodeErr := json.NewDecoder(io.LimitReader(response.Body, 4096)).Decode(&payload); decodeErr != nil {
		return fmt.Errorf("decode Upgrade refusal: %w", decodeErr)
	}
	if payload.Error.Code != "request_too_large" {
		return fmt.Errorf("h2c upgrade error code = %q, want request_too_large", payload.Error.Code)
	}
	return nil
}
