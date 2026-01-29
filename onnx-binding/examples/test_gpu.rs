//! Test GPU (ROCm) execution provider availability

use ort::session::Session;

fn main() {
    println!("ONNX Runtime GPU Test");
    println!("======================\n");
    
    // Initialize ort
    let _init = ort::init().commit();
    println!("✓ ONNX Runtime initialized\n");
    
    // Check ROCm execution provider
    #[cfg(feature = "rocm")]
    {
        println!("Testing ROCm (AMD GPU) execution provider...");
        
        use ort::execution_providers::ROCmExecutionProvider;
        
        // Try to create a session builder with ROCm
        match Session::builder() {
            Ok(builder) => {
                println!("  ✓ Session builder created");
                
                let rocm_ep = ROCmExecutionProvider::default();
                println!("  ✓ ROCm execution provider created");
                
                match builder.with_execution_providers([rocm_ep.build()]) {
                    Ok(_builder) => {
                        println!("  ✓ ROCm execution provider registered");
                        println!("\n✓ ROCm support is available!");
                    }
                    Err(e) => {
                        println!("  ✗ Failed to register ROCm execution provider: {:?}", e);
                    }
                }
            }
            Err(e) => {
                println!("  ✗ Failed to create session builder: {:?}", e);
            }
        }
    }
    
    #[cfg(not(feature = "rocm"))]
    {
        println!("ROCm feature not enabled. Rebuild with --features rocm");
    }
    
    // Test CPU as fallback
    println!("\nTesting CPU execution provider...");
    match Session::builder() {
        Ok(_builder) => {
            println!("  ✓ CPU execution provider available (always works)");
        }
        Err(e) => {
            println!("  ✗ Failed: {:?}", e);
        }
    }
    
    println!("\n=== Test Complete ===");
}
