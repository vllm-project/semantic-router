package docker

import (
	"context"
	"fmt"
	"os"
	"os/exec"
)

// Builder handles Docker image building and loading
type Builder struct {
	Verbose bool
}

// NewBuilder creates a new Docker builder
func NewBuilder(verbose bool) *Builder {
	return &Builder{
		Verbose: verbose,
	}
}

// Build builds a Docker image
func (b *Builder) Build(ctx context.Context, opts BuildOptions) error {
	b.log("Building Docker image: %s", opts.Tag)

	args := []string{"build"}

	if opts.Dockerfile != "" {
		args = append(args, "-f", opts.Dockerfile)
	}

	args = append(args, "-t", opts.Tag)

	if opts.BuildContext == "" {
		opts.BuildContext = "."
	}
	args = append(args, opts.BuildContext)

	cmd := exec.CommandContext(ctx, "docker", args...)
	if b.Verbose {
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
	}

	if err := cmd.Run(); err != nil {
		return fmt.Errorf("failed to build image: %w", err)
	}

	b.log("Image %s built successfully", opts.Tag)
	return nil
}

// LoadToKind loads a Docker image into a Kind cluster
func (b *Builder) LoadToKind(ctx context.Context, clusterName, imageTag string) error {
	b.log("Loading image %s to Kind cluster %s", imageTag, clusterName)

	cmd := exec.CommandContext(ctx, "kind", "load", "docker-image", imageTag, "--name", clusterName)
	if b.Verbose {
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
	}

	if err := cmd.Run(); err != nil {
		return fmt.Errorf("failed to load image to Kind: %w", err)
	}

	b.log("Image %s loaded to Kind cluster %s successfully", imageTag, clusterName)
	return nil
}

// BuildAndLoad builds a Docker image and loads it into a Kind cluster
func (b *Builder) BuildAndLoad(ctx context.Context, clusterName string, opts BuildOptions) error {
	if err := b.Build(ctx, opts); err != nil {
		return err
	}

	return b.LoadToKind(ctx, clusterName, opts.Tag)
}

func (b *Builder) log(format string, args ...interface{}) {
	if b.Verbose {
		fmt.Printf("[Docker] "+format+"\n", args...)
	}
}

// BuildOptions contains options for building Docker images
type BuildOptions struct {
	// Dockerfile is the path to the Dockerfile
	Dockerfile string

	// Tag is the image tag
	Tag string

	// BuildContext is the build context directory
	BuildContext string
}
