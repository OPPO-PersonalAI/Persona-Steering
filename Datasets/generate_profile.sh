#!/bin/bash
# Fill Demographic Information for LaMP / LongLaMP JSON (calls generate_profile.py).

export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-}"
export LM_MODEL="${LM_MODEL:-gpt-4o-mini}"
export BATCH_SIZE="${BATCH_SIZE:-20}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/generate_profile.py"

# Override with --dir; no hardcoded machine paths for open-source use.
DEFAULT_BASE_PATHS=()
FALLBACK_BASE_PATH="${SCRIPT_DIR}/data"

LOG_DIR="${SCRIPT_DIR}/logs"
LOG_FILE="${LOG_DIR}/generate_profile_$(date +%Y%m%d_%H%M%S).log"

print_info() {
    echo -e "\033[0;32m[INFO]\033[0m $1"
}

print_warn() {
    echo -e "\033[0;33m[WARN]\033[0m $1"
}

print_error() {
    echo -e "\033[0;31m[ERROR]\033[0m $1"
}

print_section() {
    echo -e "\n\033[1;36m========================================\033[0m"
    echo -e "\033[1;36m$1\033[0m"
    echo -e "\033[1;36m========================================\033[0m"
}

check_dependencies() {
    print_info "Checking dependencies..."
    
    if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
        print_error "Python not found"
        exit 1
    fi
    
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    else
        PYTHON_CMD="python"
    fi
    
    if ! $PYTHON_CMD -c "import openai" 2>/dev/null; then
        print_warn "openai not found; installing..."
        $PYTHON_CMD -m pip install openai --quiet
    fi
    
    print_info "Dependencies OK"
}

check_env() {
    print_info "Checking environment..."
    
    if [ "$OPENAI_API_KEY" = "your-api-key-here" ] || [ -z "$OPENAI_API_KEY" ]; then
        print_error "OPENAI_API_KEY is not set"
        print_info "  export OPENAI_API_KEY='...'"
        exit 1
    fi
    
    print_info "Environment OK"
}

setup_logging() {
    mkdir -p "$LOG_DIR"
    print_info "Log file: $LOG_FILE"
}

process_file() {
    local file_path="$1"
    
    if [ ! -f "$file_path" ]; then
        print_error "File not found: $file_path"
        return 1
    fi
    
    print_section "File: $file_path" | tee -a "$LOG_FILE"
    print_info "Start: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
    
    file_size=$(du -h "$file_path" 2>/dev/null | cut -f1 || echo "unknown")
    print_info "Size: $file_size" | tee -a "$LOG_FILE"
    
    local total=$(python3 -c "
import json
import sys

def is_invalid_demographic_info(demo_info):
    if not demo_info:
        return True
    if isinstance(demo_info, str):
        demo_str = demo_info.strip()
        if not demo_str:
            return True
        invalid_patterns = [
            'likes N/A',
            'dislikes N/A', 
            'undefined focus',
            'lack of available publication',
            'cannot be determined',
            'inability to characterize'
        ]
        invalid_count = sum(1 for pattern in invalid_patterns if pattern.lower() in demo_str.lower())
        if invalid_count >= 2:
            return True
    return False

try:
    with open('$file_path', 'r', encoding='utf-8') as f:
        data = json.load(f)
    pending = [
        item for item in data 
        if is_invalid_demographic_info(item.get('Demographic Information'))
    ]
    missing_count = sum(1 for item in pending if not item.get('Demographic Information'))
    invalid_count = sum(1 for item in pending if item.get('Demographic Information') and is_invalid_demographic_info(item.get('Demographic Information')))
    print(f'{len(data)},{len(pending)},{missing_count},{invalid_count}')
except Exception as e:
    print(f'0,0,0,0')
    sys.stderr.write(f'Error: {e}\n')
" 2>/dev/null || echo "0,0,0,0")
    
    IFS=',' read -r total_count pending_count missing_count invalid_count <<< "$total"
    
    if [ "$pending_count" -eq 0 ]; then
        print_warn "All samples already have valid Demographic Information; skipping." | tee -a "$LOG_FILE"
        return 0
    fi
    
    print_info "Total: $total_count, pending: $pending_count (missing: $missing_count, invalid: $invalid_count)" | tee -a "$LOG_FILE"
    
    export LAMP_INPUT_FILE="$file_path"
    
    $PYTHON_CMD "$PYTHON_SCRIPT" 2>&1 | tee -a "$LOG_FILE"
    local python_exit_code=${PIPESTATUS[0]}
    
    if [ $python_exit_code -eq 0 ]; then
        print_info "Done: $file_path" | tee -a "$LOG_FILE"
        print_info "End: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
        return 0
    else
        print_error "Python failed (exit $python_exit_code): $file_path" | tee -a "$LOG_FILE"
        print_error "Stopping after API/generation failure" | tee -a "$LOG_FILE"
        return 1
    fi
}

should_process_file() {
    local file_path="$1"
    
    if [[ "$file_path" == *"/longlamp/"* ]]; then
        return 0
    fi
    
    if [[ "$file_path" == *"/LaMP4/"* ]] || \
       [[ "$file_path" == *"/LaMP5/"* ]] || \
       [[ "$file_path" == *"/LaMP7/"* ]]; then
        return 0
    fi
    
    return 1
}

process_all() {
    print_section "Batch: LaMP 4/5/7 and LongLaMP"
    
    local base_paths=()
    
    if [ $# -gt 0 ] && [ -n "$1" ]; then
        if [ ! -d "$1" ]; then
            print_error "Directory not found: $1"
            exit 1
        fi
        base_paths=("$1")
    else
        base_paths=("${DEFAULT_BASE_PATHS[@]}")
        local found_path=false
        for path in "${base_paths[@]}"; do
            if [ -d "$path" ]; then
                found_path=true
                break
            fi
        done
        if [ "$found_path" = false ] && [ -d "$FALLBACK_BASE_PATH" ]; then
            print_warn "Using fallback data path: $FALLBACK_BASE_PATH"
            base_paths=("$FALLBACK_BASE_PATH")
            found_path=true
        fi
        if [ "$found_path" = false ]; then
            print_error "No dataset root found. Use: $0 --all --dir /path/to/data"
            print_info "  Expected LaMP/LaMP4, LaMP5, LaMP7 and/or longlamp/ under that root."
            exit 1
        fi
    fi
    
    local longlamp_files=()
    local lamp_files=()
    
    for base_path in "${base_paths[@]}"; do
        if [ ! -d "$base_path" ]; then
            print_warn "Skip missing directory: $base_path"
            continue
        fi
        
        print_info "Scanning: $base_path"
        
        while IFS= read -r -d '' file; do
            if should_process_file "$file"; then
                if [[ "$file" == *"/longlamp/"* ]]; then
                    longlamp_files+=("$file")
                else
                    lamp_files+=("$file")
                fi
            fi
        done < <(find "$base_path" -type f -name "*_questions.json" -print0 2>/dev/null | sort -z)
    done
    
    local all_files=("${longlamp_files[@]}" "${lamp_files[@]}")
    
    if [ ${#all_files[@]} -eq 0 ]; then
        print_error "No *_questions.json files found"
        print_info "Searched:"
        for path in "${base_paths[@]}"; do
            print_info "  - $path"
        done
        return 1
    fi
    
    print_info "Found ${#all_files[@]} files (LongLaMP: ${#longlamp_files[@]}, LaMP 4/5/7: ${#lamp_files[@]})"
    print_info "Order: LongLaMP first, then LaMP 4/5/7"
    print_info "List:"
    if [ ${#longlamp_files[@]} -gt 0 ]; then
        print_info "  [LongLaMP]"
        for file in "${longlamp_files[@]}"; do
            print_info "    - $file"
        done
    fi
    if [ ${#lamp_files[@]} -gt 0 ]; then
        print_info "  [LaMP 4,5,7]"
        for file in "${lamp_files[@]}"; do
            print_info "    - $file"
        done
    fi
    
    local success_count=0
    local fail_count=0
    local skip_count=0
    
    for file in "${all_files[@]}"; do
        process_file "$file"
        local result=$?
        
        if tail -n 20 "$LOG_FILE" 2>/dev/null | grep -q "All samples already have valid Demographic Information; skipping" || true; then
            ((skip_count++)) || true
            print_info "Skipped so far: $skip_count"
        elif [ $result -eq 0 ]; then
            ((success_count++)) || true
            print_info "Succeeded so far: $success_count"
        else
            ((fail_count++)) || true
            print_error "Failed count: $fail_count"
            print_error "Stopping batch after failure"
            print_error "Done: $success_count ok, $skip_count skipped"
            print_error "Failed file: $file"
            break
        fi
        
        local processed=$((success_count + fail_count + skip_count)) || 0
        if [ $processed -lt ${#all_files[@]} ]; then
            print_info "Progress: $processed / ${#all_files[@]} — sleeping 2s..."
            sleep 2
        fi
    done
    
    print_section "Batch finished"
    print_info "OK: $success_count  Fail: $fail_count  Skip: $skip_count"
    print_info "Log: $LOG_FILE"
}

show_help() {
    cat << EOF
Usage: $0 [options] [file]

Fill Demographic Information for LaMP / LongLaMP JSON.

Options:
    -h, --help          This help
    -a, --all           Discover LaMP 4/5/7 and LongLaMP *_questions.json
    -d, --dir DIR       Root directory for --all
    -f, --file FILE     Single JSON file
    -l, --log LOG       Log file path
    --batch-size N
    --model MODEL
    --base-url URL

Env:
    OPENAI_API_KEY (required)
    OPENAI_BASE_URL  LM_MODEL  BATCH_SIZE

Examples:
    $0 -f /path/to/dev_questions.json
    $0 --all
    $0 --all --dir /data/LaMP

Notes:
    With --all you must pass --dir <data_root> unless Datasets/data/ exists locally.
    Only LaMP4, LaMP5, LaMP7, and longlamp paths are processed.
    Files with valid demographics are skipped.

EOF
}

main() {
    print_section "LaMP Demographic Information"
    print_info "Start: $(date '+%Y-%m-%d %H:%M:%S')"
    
    local process_all_files=false
    local input_file=""
    local data_dir=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -a|--all)
                process_all_files=true
                shift
                ;;
            -d|--dir)
                data_dir="$2"
                shift 2
                ;;
            -f|--file)
                input_file="$2"
                shift 2
                ;;
            -l|--log)
                LOG_FILE="$2"
                shift 2
                ;;
            --batch-size)
                export BATCH_SIZE="$2"
                shift 2
                ;;
            --model)
                export LM_MODEL="$2"
                shift 2
                ;;
            --base-url)
                export OPENAI_BASE_URL="$2"
                shift 2
                ;;
            *)
                if [ -z "$input_file" ] && [ -f "$1" ]; then
                    input_file="$1"
                else
                    print_error "Unknown argument: $1"
                    show_help
                    exit 1
                fi
                shift
                ;;
        esac
    done
    
    check_dependencies
    check_env
    setup_logging
    
    if [ "$process_all_files" = true ]; then
        if [ -n "$data_dir" ]; then
            process_all "$data_dir"
        else
            process_all
        fi
    elif [ -n "$input_file" ]; then
        process_file "$input_file"
    else
        print_error "Pass a file or use --all"
        show_help
        exit 1
    fi
    
    print_section "Done"
    print_info "End: $(date '+%Y-%m-%d %H:%M:%S')"
    print_info "Log: $LOG_FILE"
}

main "$@"
