package main

import (
    "fmt"
    "log"

    cfg "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
    svc "github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

func main() {
    configPath := "config/config.yaml"


    routerCfg, err := cfg.ParseConfigFile(configPath)
    if err != nil {
        log.Fatalf("Failed to load config: %v", err)
    }
		fmt.Println("Loaded model path:", routerCfg.Classifier.CategoryModel.ModelID)

    // Use auto-discovery: this loads your HF BERT classifier
    service, err := svc.NewClassificationServiceWithAutoDiscovery(routerCfg)
    if err != nil {
        log.Fatalf("Failed to create classification service: %v", err)
    }
		fmt.Printf("Service created: %+v\n", service)


    req := svc.IntentRequest{
        Text: "generate an image of a castle at night",
        Options: &svc.IntentOptions{
            ReturnProbabilities: true,
        },
    }

    // IMPORTANT: CALL LEGACY CLASSIFIER, NOT UNIFIED
    resp, err := service.ClassifyIntent(req)
    if err != nil {
        log.Fatalf("Classification failed: %v", err)
    }

    fmt.Println("=== TEST RESULT ===")
    fmt.Println("Text:", req.Text)
    fmt.Println("Category:", resp.Classification.Category)
    fmt.Println("Confidence:", resp.Classification.Confidence)
    fmt.Println("Processing time (ms):", resp.Classification.ProcessingTimeMs)
    fmt.Println("RoutingDecision:", resp.RoutingDecision)
    fmt.Printf("Probabilities: %+v\n", resp.Probabilities)
}
