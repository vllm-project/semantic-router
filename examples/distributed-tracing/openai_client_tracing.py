#!/usr/bin/env python3
"""
OpenTelemetry Distributed Tracing with OpenAI Client and Semantic Router

This example demonstrates end-to-end distributed tracing from a client application
through the semantic router to vLLM backends using OpenTelemetry.

The example shows:
1. Auto-instrumentation of OpenAI Python client
2. Auto-instrumentation of HTTP requests library
3. Trace context propagation across service boundaries
4. Span attributes for debugging and analysis

Prerequisites:
- Semantic Router running with tracing enabled
- Jaeger or another OTLP collector running
- Python packages: openai, opentelemetry-* (see requirements.txt)

Signed-off-by: GitHub Copilot <noreply@github.com>
"""

import os
import sys
from typing import Optional

from openai import OpenAI
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


def setup_tracing(
    service_name: str = "openai-client-example",
    otlp_endpoint: str = "http://localhost:4317",
) -> None:
    """
    Initialize OpenTelemetry tracing with OTLP exporter.

    Args:
        service_name: Name of the service for trace identification
        otlp_endpoint: OTLP collector endpoint (e.g., Jaeger, Tempo)
    """
    # Create a resource that identifies this service
    resource = Resource(attributes={SERVICE_NAME: service_name})

    # Create tracer provider with the resource
    provider = TracerProvider(resource=resource)

    # Create OTLP exporter that sends spans to the collector
    otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)

    # Add batch processor for efficient span export
    provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

    # Set the global tracer provider
    trace.set_tracer_provider(provider)

    # Auto-instrument OpenAI client for automatic span creation
    OpenAIInstrumentor().instrument()

    # Auto-instrument requests library for HTTP trace header injection
    RequestsInstrumentor().instrument()

    print(f"✅ OpenTelemetry tracing initialized")
    print(f"   Service: {service_name}")
    print(f"   OTLP Endpoint: {otlp_endpoint}")


def main():
    """
    Main example demonstrating distributed tracing with semantic router.
    """
    # Configuration from environment variables with defaults
    router_url = os.getenv("SEMANTIC_ROUTER_URL", "http://localhost:8000")
    otlp_endpoint = os.getenv("OTLP_ENDPOINT", "http://localhost:4317")
    api_key = os.getenv("OPENAI_API_KEY", "dummy-key-for-local-testing")

    print("=" * 80)
    print("OpenTelemetry Distributed Tracing Example")
    print("=" * 80)
    print(f"Router URL: {router_url}")
    print(f"OTLP Endpoint: {otlp_endpoint}")
    print()

    # Setup OpenTelemetry tracing
    setup_tracing(
        service_name="openai-client-example", otlp_endpoint=otlp_endpoint
    )

    # Create OpenAI client pointing to semantic router
    client = OpenAI(
        base_url=f"{router_url}/v1",
        api_key=api_key,
    )

    # Get tracer for creating custom spans
    tracer = trace.get_tracer(__name__)

    try:
        # Example 1: Simple completion with auto-routing
        print("\n📝 Example 1: Auto-routing (model='auto')")
        print("-" * 80)

        with tracer.start_as_current_span("example_1_auto_routing") as span:
            # Add custom attributes to the span
            span.set_attribute("example.type", "auto_routing")
            span.set_attribute("example.number", 1)

            response = client.chat.completions.create(
                model="auto",  # Triggers semantic routing
                messages=[
                    {
                        "role": "user",
                        "content": "What is quantum computing? Explain in simple terms.",
                    }
                ],
                max_tokens=150,
            )

            print(f"Model used: {response.model}")
            print(f"Response: {response.choices[0].message.content[:200]}...")

        # Example 2: Math/reasoning query
        print("\n📝 Example 2: Math/Reasoning Query")
        print("-" * 80)

        with tracer.start_as_current_span("example_2_math_reasoning") as span:
            span.set_attribute("example.type", "math_reasoning")
            span.set_attribute("example.number", 2)

            response = client.chat.completions.create(
                model="auto",
                messages=[
                    {
                        "role": "user",
                        "content": "Calculate the compound interest on $10,000 at 5% annual rate over 3 years.",
                    }
                ],
                max_tokens=150,
            )

            print(f"Model used: {response.model}")
            print(f"Response: {response.choices[0].message.content[:200]}...")

        # Example 3: Streaming response
        print("\n📝 Example 3: Streaming Response")
        print("-" * 80)

        with tracer.start_as_current_span("example_3_streaming") as span:
            span.set_attribute("example.type", "streaming")
            span.set_attribute("example.number", 3)

            stream = client.chat.completions.create(
                model="auto",
                messages=[
                    {
                        "role": "user",
                        "content": "Write a haiku about distributed tracing.",
                    }
                ],
                max_tokens=50,
                stream=True,
            )

            print("Streaming response: ", end="", flush=True)
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    print(chunk.choices[0].delta.content, end="", flush=True)
            print()

        print("\n" + "=" * 80)
        print("✅ Examples completed successfully!")
        print("=" * 80)
        print("\n📊 View traces in Jaeger UI:")
        print("   http://localhost:16686")
        print("\n🔍 Search for service: 'openai-client-example'")
        print("   You should see traces with spans from:")
        print("   - openai-client-example (this application)")
        print("   - vllm-semantic-router (the router)")
        print("   - Any configured vLLM backends (if instrumented)")

    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        print(
            "\nTroubleshooting:",
            file=sys.stderr,
        )
        print(f"  1. Ensure semantic router is running at {router_url}", file=sys.stderr)
        print(f"  2. Ensure OTLP collector is running at {otlp_endpoint}", file=sys.stderr)
        print(
            "  3. Check router config has tracing enabled",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
