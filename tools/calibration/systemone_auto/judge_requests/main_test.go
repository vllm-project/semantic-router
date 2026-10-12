package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"
)

type fixture struct {
	Source        json.RawMessage `json:"source"`
	Model         string          `json:"served_model_id"`
	Request       json.RawMessage `json:"request"`
	ContentSHA256 []string        `json:"message_content_sha256"`
}

func loadFixture(t *testing.T) fixture {
	t.Helper()
	_, file, _, _ := runtime.Caller(0)
	body, err := os.ReadFile(filepath.Join(filepath.Dir(file), "../tests/fixtures/judge-request.json"))
	if err != nil {
		t.Fatal(err)
	}
	var value fixture
	if err := json.Unmarshal(body, &value); err != nil {
		t.Fatal(err)
	}
	return value
}

func TestServingPromptGolden(t *testing.T) {
	golden := loadFixture(t)
	var output bytes.Buffer
	if err := generate(bytes.NewReader(golden.Source), &output, golden.Model); err != nil {
		t.Fatal(err)
	}
	var generated outputRecord
	if err := json.Unmarshal(output.Bytes(), &generated); err != nil {
		t.Fatal(err)
	}
	var actual, expected any
	if err := json.Unmarshal(generated.Request, &actual); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(golden.Request, &expected); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(actual, expected) {
		t.Fatal("production judge request differs from shared golden")
	}
	var request struct {
		Messages []struct {
			Content string `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(generated.Request, &request); err != nil {
		t.Fatal(err)
	}
	for i, message := range request.Messages {
		sum := sha256.Sum256([]byte(message.Content))
		if hex.EncodeToString(sum[:]) != golden.ContentSHA256[i] {
			t.Fatal("judge prompt UTF-8 bytes changed")
		}
	}
}

func TestRejectsLabelledOrDuplicateInputs(t *testing.T) {
	golden := loadFixture(t)
	labelled := strings.TrimSuffix(strings.TrimSpace(string(golden.Source)), "}") + `,"labels":{"answer":"yes"}}`
	for name, input := range map[string]string{"labels": labelled, "duplicate": string(golden.Source) + "\n" + string(golden.Source)} {
		t.Run(name, func(t *testing.T) {
			var output bytes.Buffer
			if err := generate(strings.NewReader(input), &output, golden.Model); err == nil {
				t.Fatal("invalid input was accepted")
			}
		})
	}
}
