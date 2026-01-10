#!/bin/bash

################################################################################
# Master Script - Analisi Completa Block Size per Game of Life
# 
# Questo script esegue:
# 1. Benchmark di tutte le configurazioni
# 2. Analisi dei risultati
################################################################################

# Colori
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# Banner
show_banner() {
    clear
    echo -e "${CYAN}"
    echo "╔════════════════════════════════════════════════════════════════════╗"
    echo "║                                                                    ║"
    echo "║        🎯 CUDA BLOCK SIZE OPTIMIZATION ANALYZER 🎯                ║"
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
    
    # Check nvcc
    if ! command -v nvcc &> /dev/null; then
        echo -e "${RED}  ✗ nvcc not found - CUDA toolkit required${NC}"
        all_ok=false
    else
        echo -e "${GREEN}  ✓ nvcc found${NC}"
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
    echo -e "${YELLOW}  This will take approximately 5-10 minutes${NC}\n"
    
    # Configuration
    BLOCK_SIZES=(1 4 8 16 32)
    GRID_SIZES=(256 512 1024 2048)
    GENERATIONS=100
    
    # Setup directories
    mkdir -p build/block_tests
    mkdir -p benchmarks
    
    # Results file
    RESULTS_FILE="benchmarks/block_size_comparison.csv"
    echo "block_size,grid_size,generations,time_ms,time_per_gen_ms,throughput_mcells_s" > $RESULTS_FILE
    
    # Compile all versions
    echo -e "${CYAN}  Compiling CUDA kernels...${NC}"
    for BS in "${BLOCK_SIZES[@]}"; do
        printf "    BS=%2d: " $BS
        nvcc -o build/block_tests/game_of_life_bs${BS} \
             -DBLOCK_SIZE=${BS} \
             src/cuda/game_of_life.cu \
             -O3 2>&1 | grep -q "error"
        
        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            echo -e "${GREEN}✓${NC}"
        else
            echo -e "${RED}✗${NC}"
            exit 1
        fi
    done
    
    echo ""
    
    # Run benchmarks
    total_tests=$((${#BLOCK_SIZES[@]} * ${#GRID_SIZES[@]}))
    current_test=0
    
    for GRID in "${GRID_SIZES[@]}"; do
        echo -e "${CYAN}  Grid ${GRID}×${GRID}:${NC}"
        
        for BS in "${BLOCK_SIZES[@]}"; do
            current_test=$((current_test + 1))
            progress=$((current_test * 100 / total_tests))
            
            printf "    BS=%2d [%3d%%]: " $BS $progress
            
            # Run and capture output
            output=$(./build/block_tests/game_of_life_bs${BS} ${GRID} ${GRID} ${GENERATIONS} 0 42 2>&1)
            
            # Extract metrics
            time_ms=$(echo "$output" | grep "Total time:" | awk '{print $3}')
            
            if [ -n "$time_ms" ]; then
                time_per_gen=$(echo "scale=6; $time_ms / $GENERATIONS" | bc)
                throughput=$(echo "scale=4; $GRID * $GRID * $GENERATIONS / $time_ms / 1000" | bc)
                
                echo "${BS},${GRID},${GENERATIONS},${time_ms},${time_per_gen},${throughput}" >> $RESULTS_FILE
                
                echo -e "${GREEN}${time_ms} ms (${throughput} M cells/s)${NC}"
            else
                echo -e "${RED}Failed${NC}"
            fi
        done
        echo ""
    done
    
    echo -e "${GREEN}  ✓ Benchmarks completed!${NC}\n"
}

# Show results
show_results() {
    echo -e "${BLUE}[3/3] Analysis complete!${NC}\n"
    
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC}                     ${GREEN}📊 RESULTS SUMMARY${NC}                        ${CYAN}║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}\n"
    
    # Parse best configuration from CSV
    if [ -f "benchmarks/block_size_comparison.csv" ]; then
        best_line=$(tail -n +2 benchmarks/block_size_comparison.csv | \
                    sort -t',' -k6 -nr | head -n1)
        
        best_bs=$(echo $best_line | cut -d',' -f1)
        best_grid=$(echo $best_line | cut -d',' -f2)
        best_throughput=$(echo $best_line | cut -d',' -f6)
        
        echo -e "${GREEN}🏆 OPTIMAL CONFIGURATION:${NC}"
        echo -e "   Block Size: ${YELLOW}${best_bs}×${best_bs}${NC} (${best_bs}² = $((best_bs * best_bs)) threads/block)"
        echo -e "   Peak Performance: ${YELLOW}${best_throughput} M cells/s${NC}"
        echo -e "   Best Grid Size: ${YELLOW}${best_grid}×${best_grid}${NC}\n"
    fi
    
    echo -e "${CYAN}📂 Generated Files:${NC}"
    echo ""
    
    if [ -f "benchmarks/block_size_comparison.csv" ]; then
        size=$(du -h "benchmarks/block_size_comparison.csv" | cut -f1)
        echo -e "   ${GREEN}✓${NC} 📊 Raw benchmark data"
        echo -e "     ${BLUE}→${NC} benchmarks/block_size_comparison.csv ${YELLOW}($size)${NC}"
    fi
    
    echo ""
    echo -e "${CYAN}📊 View results with:${NC}"
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
    echo -e "${CYAN}║${NC}                       ${GREEN}✓ BENCHMARK COMPLETE!${NC}                   ${CYAN}║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}\n"
    
    echo -e "${YELLOW}💡 Analysis Tips:${NC}\n"
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
    
    # Full pipeline
    check_prerequisites
    run_benchmarks
    show_results
    view_results
    final_message
}

# Trap errors
trap 'echo -e "\n${RED}Script interrupted!${NC}\n"; exit 1' INT TERM

# Run main
main "$@"