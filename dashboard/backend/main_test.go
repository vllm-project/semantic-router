package main

import (
	"context"
	"errors"
	"net/http"
	"reflect"
	"testing"
	"time"
)

func TestRunServerLifecycleGracefullyShutsDownBeforeClosingResources(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	served := make(chan struct{})
	release := make(chan struct{})
	steps := make([]string, 0, 2)
	done := make(chan error, 1)
	go func() {
		done <- runServerLifecycle(ctx, func() error {
			close(served)
			<-release
			return http.ErrServerClosed
		}, func(context.Context) error {
			steps = append(steps, "shutdown")
			close(release)
			return nil
		}, func() error {
			steps = append(steps, "close")
			return nil
		}, time.Second)
	}()
	<-served
	cancel()
	if err := <-done; err != nil {
		t.Fatalf("runServerLifecycle: %v", err)
	}
	if !reflect.DeepEqual(steps, []string{"shutdown", "close"}) {
		t.Fatalf("lifecycle order=%v", steps)
	}
}

func TestRunServerLifecycleJoinsShutdownServeAndCloseErrors(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	serveStarted := make(chan struct{})
	release := make(chan struct{})
	serveErr, shutdownErr, closeErr := errors.New("serve"), errors.New("shutdown"), errors.New("close")
	done := make(chan error, 1)
	go func() {
		done <- runServerLifecycle(ctx, func() error {
			close(serveStarted)
			<-release
			return serveErr
		}, func(context.Context) error {
			close(release)
			return shutdownErr
		}, func() error { return closeErr }, time.Second)
	}()
	<-serveStarted
	cancel()
	err := <-done
	for _, expected := range []error{serveErr, shutdownErr, closeErr} {
		if !errors.Is(err, expected) {
			t.Fatalf("joined error %v does not contain %v", err, expected)
		}
	}
}
