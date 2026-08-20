package auth

import (
	"context"
	"fmt"
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

func (m SMTPInvitationMailer) SendDashboardInvitation(_ context.Context, recipient, invitationURL string, expiresAt time.Time) error {
	if !m.Configured() {
		return fmt.Errorf("SMTP invitation delivery is not configured")
	}
	address := m.Host + ":" + strconv.Itoa(m.Port)
	var auth smtp.Auth
	if m.Username != "" {
		auth = smtp.PlainAuth("", m.Username, m.Password, m.Host)
	}
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
	return smtp.SendMail(address, auth, m.From, []string{recipient}, []byte(body))
}
