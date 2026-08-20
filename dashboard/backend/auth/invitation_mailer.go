package auth

import (
	"context"
	"crypto/tls"
	"fmt"
	"net"
	"net/smtp"
	"strconv"
	"strings"
	"time"
)

type SMTPInvitationMailer struct {
	Host     string
	Port     int
	Username string
	Password string
	From     string
}

func (m SMTPInvitationMailer) Configured() bool {
	return strings.TrimSpace(m.Host) != "" && m.Port > 0 && strings.TrimSpace(m.From) != ""
}

func (m SMTPInvitationMailer) SendDashboardInvitation(ctx context.Context, recipient, invitationURL string, expiresAt time.Time) error {
	if !m.Configured() {
		return fmt.Errorf("SMTP invitation delivery is not configured")
	}
	address := m.Host + ":" + strconv.Itoa(m.Port)
	body := strings.Join([]string{
		"From: " + m.From,
		"To: " + recipient,
		"Subject: You are invited to the vLLM Semantic Router Dashboard",
		"MIME-Version: 1.0",
		"Content-Type: text/plain; charset=UTF-8",
		"",
		"An administrator invited you to the vLLM Semantic Router Dashboard.",
		"",
		"Accept the invitation: " + invitationURL,
		"",
		"This one-time link expires at " + expiresAt.UTC().Format(time.RFC3339) + ".",
	}, "\r\n")
	client, err := m.secureClient(ctx, address)
	if err != nil {
		return err
	}
	defer client.Close()
	if m.Username != "" {
		if authErr := client.Auth(smtp.PlainAuth("", m.Username, m.Password, m.Host)); authErr != nil {
			return fmt.Errorf("authenticate with SMTP server: %w", authErr)
		}
	}
	if senderErr := client.Mail(m.From); senderErr != nil {
		return fmt.Errorf("set invitation sender: %w", senderErr)
	}
	if recipientErr := client.Rcpt(recipient); recipientErr != nil {
		return fmt.Errorf("set invitation recipient: %w", recipientErr)
	}
	writer, err := client.Data()
	if err != nil {
		return fmt.Errorf("start invitation message: %w", err)
	}
	if _, writeErr := writer.Write([]byte(body)); writeErr != nil {
		_ = writer.Close()
		return fmt.Errorf("write invitation message: %w", writeErr)
	}
	if closeErr := writer.Close(); closeErr != nil {
		return fmt.Errorf("send invitation message: %w", closeErr)
	}
	if quitErr := client.Quit(); quitErr != nil {
		return fmt.Errorf("close SMTP session: %w", quitErr)
	}
	return nil
}

func (m SMTPInvitationMailer) secureClient(ctx context.Context, address string) (*smtp.Client, error) {
	tlsConfig := &tls.Config{MinVersion: tls.VersionTLS12, ServerName: m.Host}
	dialer := &net.Dialer{}
	connection, err := dialer.DialContext(ctx, "tcp", address)
	if err != nil {
		return nil, fmt.Errorf("connect to SMTP server: %w", err)
	}
	if deadline, ok := ctx.Deadline(); ok {
		if deadlineErr := connection.SetDeadline(deadline); deadlineErr != nil {
			connection.Close()
			return nil, fmt.Errorf("set SMTP deadline: %w", deadlineErr)
		}
	}
	if m.Port == 465 {
		secureConnection := tls.Client(connection, tlsConfig)
		if handshakeErr := secureConnection.HandshakeContext(ctx); handshakeErr != nil {
			connection.Close()
			return nil, fmt.Errorf("secure implicit TLS SMTP session: %w", handshakeErr)
		}
		connection = secureConnection
	}
	client, err := smtp.NewClient(connection, m.Host)
	if err != nil {
		connection.Close()
		return nil, fmt.Errorf("open SMTP session: %w", err)
	}
	if m.Port == 465 {
		return client, nil
	}
	if supported, _ := client.Extension("STARTTLS"); !supported {
		client.Close()
		return nil, fmt.Errorf("SMTP server must support STARTTLS")
	}
	if startTLSErr := client.StartTLS(tlsConfig); startTLSErr != nil {
		client.Close()
		return nil, fmt.Errorf("secure SMTP session with STARTTLS: %w", startTLSErr)
	}
	return client, nil
}
