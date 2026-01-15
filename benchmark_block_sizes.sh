#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
NC='\033[0m'

show_banner() {
    clear
    echo -e "${CYAN}"
    echo "╔════════════════════════════════════════════════════════════════════╗"
    echo "║                                                                    ║"
    echo "║        CUDA BLOCK SIZE OPTIMIZATION ANALYZER                 ║"
    echo "║                                                                    ║"
    echo "║              Conway's Game of Life - Full Analysis                ║"
    echo "║                                                                    ║"
    echo "╚════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}\n"
}

# Check prerequisites
check_prerequisites() {
    echo -e "${BLUE}[1/3] Checking prerequisites...${NC}"
    
    local all_ok=true
    
    if ! command -v nvcc &> /dev/null; then
        echo -e "${RED}  nvcc not found - CUDA toolkit required${NC}"
        all_ok=false
    else
        echo -e "${GREEN}  nvcc found${NC}"
    fi
    
    if [ "$all_ok" = false ]; then
        echo -e "\n${RED}Prerequisites check failed. Please install CUDA toolkit.${NC}\n"
        exit 1
    fi
    
    echo -e "${GREEN}  All prerequisites satisfied!${NC}\n"
}

# Run benchmarks
run_benchmarks() {
    echo -e "${BLUE}[2/3] Running benchmarks...${NC}"
    echo -e "${YELLOW}  This will take approximately 15-20 minutes (with averaging)${NC}\n"
    
    BLOCK_SIZES=(1 4 8 16 32)
    GRID_SIZES=(256 512 1024 2048)
    GENERATIONS=100
    WARMUP_RUNS=2
    MEASURE_RUNS=10
    
    mkdir -p build/block_tests
    mkdir -p benchmarks
    
    RESULTS_FILE="benchmarks/block_size_comparison.csv"
    echo "block_size,grid_size,generations,runs,mean_time_ms,std_time_ms,median_time_ms,min_time_ms,max_time_ms,mean_throughput_mcells_s,std_throughput_mcells_s,cv_percent" > $RESULTS_FILE
    
    echo -e "${CYAN}  Compiling CUDA kernels...${NC}"
    for BS in "${BLOCK_SIZES[@]}"; do
        printf "    BS=%2d: " $BS
        nvcc -o build/block_tests/game_of_life_bs${BS} \
             -DBLOCK_SIZE=${BS} \
             src/cuda/game_of_life.cu \
             -O3 2>&1 | grep -q "error"
        
        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            echo -e "${GREEN}Compiled${NC}"
        else
            echo -e "${RED}Failed${NC}"
            exit 1
        fi
    done
    
    echo ""
    
    total_tests=$((${#BLOCK_SIZES[@]} * ${#GRID_SIZES[@]}))
    current_test=0
    
    for GRID in "${GRID_SIZES[@]}"; do
        echo -e "${CYAN}  Grid ${GRID}×${GRID}:${NC}"
        
        for BS in "${BLOCK_SIZES[@]}"; do
            current_test=$((current_test + 1))
            progress=$((current_test * 100 / total_tests))
            
            printf "    BS=%2d [%3d%%]: " $BS $progress
            
            declare -a times
            
            for ((w=0; w<WARMUP_RUNS; w++)); do
                ./build/block_tests/game_of_life_bs${BS} ${GRID} ${GRID} ${GENERATIONS} 0 42 &>/dev/null
            done
            
            for ((m=0; m<MEASURE_RUNS; m++)); do
                output=$(./build/block_tests/game_of_life_bs${BS} ${GRID} ${GRID} ${GENERATIONS} 0 42 2>&1)
                time_ms=$(echo "$output" | grep "Total time:" | awk '{print $3}')
                times+=($time_ms)
            done
            
            stats=$(python3 -c "
import sys
import math

times = [${times[*]}]
n = len(times)

mean = sum(times) / n
variance = sum((x - mean) ** 2 for x in times) / (n - 1)
std_dev = math.sqrt(variance)
median = sorted(times)[n // 2] if n % 2 else (sorted(times)[n//2-1] + sorted(times)[n//2]) / 2
min_time = min(times)
max_time = max(times)
cv = (std_dev / mean * 100) if mean > 0 else 0

throughput = ${GRID} * ${GRID} * ${GENERATIONS} / mean / 1000
std_throughput = throughput * (std_dev / mean) if mean > 0 else 0

print(f'{mean:.4f},{std_dev:.4f},{median:.4f},{min_time:.4f},{max_time:.4f},{throughput:.4f},{std_throughput:.4f},{cv:.4f}')
")
            
            if [ -n "$stats" ]; then
                echo "${BS},${GRID},${GENERATIONS},${MEASURE_RUNS},${stats}" >> $RESULTS_FILE
                
                mean_time=$(echo $stats | cut -d',' -f1)
                throughput=$(echo $stats | cut -d',' -f6)
                cv=$(echo $stats | cut -d',' -f8)
                
                echo -e "${GREEN}${mean_time} ms (${throughput} M cells/s, CV ${cv}%)${NC}"
            else
                echo -e "${RED}Failed${NC}"
            fi
            
            unset times
        done
        echo ""
    done
    
    echo -e "${GREEN}  Benchmarks completed!${NC}\n"
}

# Show results
show_results() {
    echo -e "${BLUE}[3/3] Analysis complete!${NC}\n"
    
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC}                     ${GREEN}RESULTS SUMMARY${NC}                        ${CYAN}║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}\n"
    
    if [ -f "benchmarks/block_size_comparison.csv" ]; then
        best_line=$(tail -n +2 benchmarks/block_size_comparison.csv | \
                    sort -t',' -k6 -nr | head -n1)
        
        best_bs=$(echo $best_line | cut -d',' -f1)
        best_grid=$(echo $best_line | cut -d',' -f2)
        best_throughput=$(echo $best_line | cut -d',' -f6)
        
        echo -e "${GREEN}OPTIMAL CONFIGURATION:${NC}"
        echo -e "   Block Size: ${YELLOW}${best_bs}×${best_bs}${NC} (${best_bs}² = $((best_bs * best_bs)) threads/block)"
        echo -e "   Peak Performance: ${YELLOW}${best_throughput} M cells/s${NC}"
        echo -e "   Best Grid Size: ${YELLOW}${best_grid}×${best_grid}${NC}\n"
    fi
    
    echo -e "${CYAN}Generated Files:${NC}"
    echo ""
    
    if [ -f "benchmarks/block_size_comparison.csv" ]; then
        size=$(du -h "benchmarks/block_size_comparison.csv" | cut -f1)
        echo -e "   ${GREEN}Raw benchmark data${NC}"
        echo -e "     ${BLUE}→${NC} benchmarks/block_size_comparison.csv ${YELLOW}($size)${NC}"
    fi
    
    echo ""
    echo -e "${CYAN}View results with:${NC}"
    echo -e "   ${YELLOW}cat benchmarks/block_size_comparison.csv | column -t -s','${NC}"
    echo ""
}

# View results
view_results() {
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC}                  ${YELLOW}Would you like to view the results?${NC}           ${CYAN}║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}\n"
    
    echo -e "  ${GREEN}1${NC}) View CSV data"
    echo -e "  ${GREEN}2${NC}) Skip"
    echo ""
    read -p "Choice [1-2]: " choice
    
    case $choice in
        1)
            cat benchmarks/block_size_comparison.csv | column -t -s','
            echo ""
            ;;
        2|*)
            echo -e "${BLUE}Skipping...${NC}"
            ;;
    esac
    
    echo ""
}

# Final message
final_message() {
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC}                       ${GREEN}BENCHMARK COMPLETE!${NC}                   ${CYAN}║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}\n"
    
    echo -e "${YELLOW}Analysis Tips:${NC}\n"
    echo -e "  • Compare throughput (M cells/s) across different block sizes"
    echo -e "  • BS=16 typically optimal for warp alignment"
    echo -e "  • Consider shared memory usage for each configuration"
    echo -e "  • Check time_per_gen_ms for performance trends\n"
    
    echo -e "${CYAN}Results saved in: ${GREEN}benchmarks/block_size_comparison.csv${NC}\n"
}

################################################################################
# MAIN EXECUTION
################################################################################

main() {
    show_banner
    
    # Check if user wants to skip benchmarks
    if [ -f "benchmarks/block_size_comparison.csv" ]; then
        echo -e "${YELLOW}Existing benchmark data found.${NC}"
        read -p "Run benchmarks again? [y/N]: " run_bench
        if [[ ! $run_bench =~ ^[Yy]$ ]]; then
            echo -e "${BLUE}Skipping benchmarks, using existing data...${NC}\n"
            show_results
            view_results
            final_message
            exit 0
        fi
    fi
    
    check_prerequisites
    run_benchmarks
    show_results
    view_results
    final_message
}

trap 'echo -e "\n${RED}Script interrupted!${NC}\n"; exit 1' INT TERM

main "$@"