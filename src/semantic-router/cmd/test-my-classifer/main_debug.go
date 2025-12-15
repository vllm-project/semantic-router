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
    
    fmt.Printf("Config loaded:\n")
    fmt.Printf("  ModelID: %s\n", routerCfg.Classifier.CategoryModel.ModelID)
    fmt.Printf("  UseCPU: %v\n", routerCfg.Classifier.CategoryModel.UseCPU)
    fmt.Printf("  UseModernBERT: %v\n", routerCfg.Classifier.CategoryModel.UseModernBERT)
    fmt.Printf("  Mapping path: %s\n", routerCfg.Classifier.CategoryModel.CategoryMappingPath)
    
    service, err := svc.NewClassificationServiceWithAutoDiscovery(routerCfg)
    if err != nil {
        log.Fatalf("Failed to create classification service: %v", err)
    }
    
    fmt.Printf("\nService created successfully\n")
    
    req := svc.IntentRequest{
        Text: "generate an image of a castle at night",
        Options: &svc.IntentOptions{
            ReturnProbabilities: true,
        },
    }
    
    resp, err := service.ClassifyIntent(req)
    if err != nil {
        log.Fatalf("Classification failed: %v", err)
    }
    
    fmt.Println("\n=== TEST RESULT ===")
    fmt.Println("Text:", req.Text)
    fmt.Println("Category:", resp.Classification.Category)
    fmt.Println("Confidence:", resp.Classification.Confidence)
    fmt.Printf("Probabilities: %+v\n", resp.Probabilities)
}
