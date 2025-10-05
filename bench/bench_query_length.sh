#!/bin/bash


# Purpose:
#   This script sends HTTP requests with different content lengths to the 
#   Semantic Router (Envoy proxy) and measures the response times. It's designed
#   to help analyze how query length affects inference performance and latency.
#
# Usage:
#   The script sends serial curl requests with incrementally increasing content
#   lengths (configurable start, end, and step size) and records:
#   - Request character count
#   - Response time (client send to response received)
#   - HTTP status codes
#   - Full response content
#
# Integration with Metrics:
#   This script can be used alongside metrics collection systems to correlate
#   client-side response times with server-side inference metrics, helping to
#   understand the relationship between query length and processing time.

# Default configuration
DEFAULT_PORT=8801
DEFAULT_START_LENGTH=100
DEFAULT_END_LENGTH=2000
DEFAULT_STEP=200
DEFAULT_HOST="127.0.0.1"
DEFAULT_MODEL="auto"
DEFAULT_SLEEP=1

# Parse command line arguments
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -p, --port PORT          Envoy proxy port (default: $DEFAULT_PORT)"
    echo "  -h, --host HOST          Envoy proxy host (default: $DEFAULT_HOST)"
    echo "  -s, --start START        Starting character length (default: $DEFAULT_START_LENGTH)"
    echo "  -e, --end END            Ending character length (default: $DEFAULT_END_LENGTH)"
    echo "  -t, --step STEP          Step size for character increment (default: $DEFAULT_STEP)"
    echo "  -m, --model MODEL        Model name to use (default: $DEFAULT_MODEL)"
    echo "  -w, --wait SECONDS       Wait time between requests (default: $DEFAULT_SLEEP)"
    echo "  --help                   Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Use all defaults"
    echo "  $0 -p 8080 -s 500 -e 5000 -t 500    # Custom port, range 500-5000, step 500"
    echo "  $0 --host 192.168.1.100 --port 8801  # Custom host and port"
    echo ""
}

# Initialize variables with defaults
PORT=$DEFAULT_PORT
HOST=$DEFAULT_HOST
START_LENGTH=$DEFAULT_START_LENGTH
END_LENGTH=$DEFAULT_END_LENGTH
STEP=$DEFAULT_STEP
MODEL=$DEFAULT_MODEL
SLEEP_TIME=$DEFAULT_SLEEP

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -h|--host)
            HOST="$2"
            shift 2
            ;;
        -s|--start)
            START_LENGTH="$2"
            shift 2
            ;;
        -e|--end)
            END_LENGTH="$2"
            shift 2
            ;;
        -t|--step)
            STEP="$2"
            shift 2
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -w|--wait)
            SLEEP_TIME="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate parameters
if ! [[ "$PORT" =~ ^[0-9]+$ ]] || [ "$PORT" -lt 1 ] || [ "$PORT" -gt 65535 ]; then
    echo "Error: Port must be a number between 1 and 65535"
    exit 1
fi

if ! [[ "$START_LENGTH" =~ ^[0-9]+$ ]] || [ "$START_LENGTH" -lt 1 ]; then
    echo "Error: Start length must be a positive number"
    exit 1
fi

if ! [[ "$END_LENGTH" =~ ^[0-9]+$ ]] || [ "$END_LENGTH" -lt "$START_LENGTH" ]; then
    echo "Error: End length must be a positive number greater than or equal to start length"
    exit 1
fi

if ! [[ "$STEP" =~ ^[0-9]+$ ]] || [ "$STEP" -lt 1 ]; then
    echo "Error: Step must be a positive number"
    exit 1
fi

if ! [[ "$SLEEP_TIME" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo "Error: Sleep time must be a non-negative number"
    exit 1
fi

# Random word list
words=("apple" "banana" "cherry" "date" "elderberry" "fig" "grape" "honeydew" "kiwi" "lemon" "mango" "orange" "papaya" "quince" "raspberry" "strawberry" "tangerine" "watermelon" "blueberry" "blackberry" "cranberry" "pineapple" "coconut" "avocado" "peach" "pear" "plum" "apricot" "nectarine" "pomegranate" "dragonfruit" "passionfruit" "guava" "lychee" "rambutan" "durian" "jackfruit" "starfruit" "persimmon" "kumquat" "lime" "grapefruit" "clementine" "mandarin" "tangelo" "ugli" "pomelo" "yuzu" "calamansi" "bergamot")

# Function to generate random string content
generate_random_content() {
    local target_length=$1
    local content=""
    local current_length=0
    
    while [ $current_length -lt $target_length ]; do
        # Randomly select a word
        local word=${words[$RANDOM % ${#words[@]}]}
        
        # Add the word if it doesn't exceed the target length
        if [ $((current_length + ${#word} + 1)) -le $target_length ]; then
            if [ -z "$content" ]; then
                content="$word"
            else
                content="$content $word"
            fi
            current_length=${#content}
        else
            # If adding the full word would exceed length, add partial characters
            local remaining=$((target_length - current_length))
            if [ $remaining -gt 0 ]; then
                content="$content${word:0:$remaining}"
            fi
            break
        fi
    done
    
    echo "$content"
}

# Create result log file
log_file="test_results_$(date +%Y%m%d_%H%M%S).log"

echo "Starting serial testing..."
echo "Configuration:"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Model: $MODEL"
echo "  Test range: $START_LENGTH-$END_LENGTH characters, incrementing by $STEP characters each time"
echo "  Wait time between requests: ${SLEEP_TIME}s"
echo "  Results will be saved to: $log_file"
echo ""

# Record start time
start_time=$(date)
echo "Test start time: $start_time" | tee -a "$log_file"
echo "Configuration: HOST=$HOST, PORT=$PORT, MODEL=$MODEL, RANGE=$START_LENGTH-$END_LENGTH, STEP=$STEP, WAIT=${SLEEP_TIME}s" | tee -a "$log_file"
echo "===========================================" | tee -a "$log_file"

# Execute test requests serially
for length in $(seq $START_LENGTH $STEP $END_LENGTH); do
    echo ""
    echo "Testing request with $length characters..."
    echo "Time: $(date)" | tee -a "$log_file"
    echo "Character count: $length" | tee -a "$log_file"
    
    # Generate random content
    content=$(generate_random_content $length)
    
    # Send curl request and record results
    echo "Sending request..." | tee -a "$log_file"
    response=$(curl -s -X POST http://$HOST:$PORT/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d "{
        \"model\": \"$MODEL\",
        \"messages\": [
          {\"role\": \"user\", \"content\": \"$content\"}
        ]
      }" \
      -w "\nHTTP_CODE:%{http_code}\nTIME_TOTAL:%{time_total}\n")
    
    # Parse response
    http_code=$(echo "$response" | grep "HTTP_CODE:" | cut -d: -f2)
    time_total=$(echo "$response" | grep "TIME_TOTAL:" | cut -d: -f2)
    response_body=$(echo "$response" | sed '/HTTP_CODE:/d' | sed '/TIME_TOTAL:/d')
    
    # Record results
    echo "HTTP status code: $http_code" | tee -a "$log_file"
    echo "Response time: ${time_total} seconds" | tee -a "$log_file"
    echo "Response content: $response_body" | tee -a "$log_file"
    echo "-------------------------------------------" | tee -a "$log_file"
    
    # If HTTP status code is not 200, record error
    if [ "$http_code" != "200" ]; then
        echo "Warning: Request failed, HTTP status code: $http_code" | tee -a "$log_file"
    fi
    
    echo "Completed $length character test, waiting ${SLEEP_TIME} second(s) before continuing..."
    sleep $SLEEP_TIME
done

# Record end time
end_time=$(date)
echo ""
echo "===========================================" | tee -a "$log_file"
echo "Test end time: $end_time" | tee -a "$log_file"
echo "All tests completed! Results saved to: $log_file"
