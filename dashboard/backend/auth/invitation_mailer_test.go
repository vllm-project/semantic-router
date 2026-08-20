package auth

import (
	"bufio"
	"context"
	"fmt"
	"net"
	"strings"
	"testing"
	"time"
)

func TestSMTPInvitationMailerRequiresSTARTTLS(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()

	serverDone := make(chan struct{})
	go func() {
		defer close(serverDone)
		connection, acceptErr := listener.Accept()
		if acceptErr != nil {
			return
		}
		defer connection.Close()
		_, _ = fmt.Fprint(connection, "220 test ESMTP\r\n")
		scanner := bufio.NewScanner(connection)
		for scanner.Scan() {
			if strings.HasPrefix(scanner.Text(), "EHLO ") {
				_, _ = fmt.Fprint(connection, "250 test\r\n")
				return
			}
		}
	}()

	address := listener.Addr().(*net.TCPAddr)
	mailer := SMTPInvitationMailer{
		Host: "127.0.0.1",
		Port: address.Port,
		From: "router@example.com",
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	err = mailer.SendDashboardInvitation(
		ctx,
		"member@example.com",
		"https://dashboard.example.com/invite/token",
		time.Now().Add(time.Hour),
	)
	if err == nil || !strings.Contains(err.Error(), "must support STARTTLS") {
		t.Fatalf("SendDashboardInvitation() error=%v, want STARTTLS requirement", err)
	}
	<-serverDone
}
