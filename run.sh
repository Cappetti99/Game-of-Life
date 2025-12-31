#!/bin/bash

# Main runner script for Conway's Game of Life project
# Supports all implementations: Sequential, CUDA, and Visual

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Directory paths
SRC_DIR="$SCRIPT_DIR/src"
BUILD_DIR="$SCRIPT_DIR/build"
BENCHMARK_DIR="$SCRIPT_DIR/benchmarks"

# Find Python command (python3 or python)
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "Error: Python not found"
    exit 1
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

show_banner() {
    echo -e "${CYAN}"
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║                                                            ║"
    echo "║              Conway's Game of Life                         ║"
    echo "║                                                            ║"
    echo "║     Sequential (Python) | Parallel (CUDA) | Visual         ║"
    echo "║                                                            ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

show_help() {
    echo -e "${YELLOW}Usage:${NC} ./run.sh [command] [options]"
    echo ""
    echo -e "${GREEN}Commands:${NC}"
    echo "  all, a           Run all implementations (sequential + CUDA)"
    echo "  visual, v        Run visual Pygame version (interactive)"
    echo "  sequential, s    Run Python sequential version"
    echo "  cuda, c          Compile and run CUDA parallel version"
    echo "  benchmark, b     Run benchmarks for all implementations"
    echo "  clean            Remove build artifacts"
    echo "  help, h          Show this help message"
    echo ""
    echo -e "${GREEN}Examples:${NC}"
    echo "  ./run.sh all                       # Run sequential and CUDA with defaults"
    echo "  ./run.sh all 512 512 100           # Run both: 512x512, 100 generations"
    echo "  ./run.sh visual                    # Run visual version with defaults"
    echo "  ./run.sh visual 200 150 5          # Visual: 200x150 grid, 5px cells"
    echo "  ./run.sh sequential 128 128 500    # Python: 128x128, 500 generations"
    echo "  ./run.sh cuda 1024 1024 1000       # CUDA: 1024x1024, 1000 generations"
    echo "  ./run.sh benchmark                 # Run all benchmarks"
    echo ""
}

run_visual() {
    echo -e "${MAGENTA}▶ Starting Visual Version (Pygame)${NC}"
    echo ""
    
    # Check if pygame is installed
    if ! $PYTHON -c "import pygame" 2>/dev/null; then
        echo -e "${YELLOW}Installing pygame...${NC}"
        $PYTHON -m pip install pygame -q
    fi
    
    echo -e "${YELLOW}Controls:${NC}"
    echo -e "  SPACE: Play/Pause | R: Random | C: Clear | G: Glider | U: Gun"
    echo -e "  LEFT CLICK: Draw | RIGHT CLICK: Erase | UP/DOWN: Speed | ESC: Quit"
    echo ""
    
    $PYTHON "$SRC_DIR/visual/game_of_life_visual.py" "$@"
}

run_sequential() {
    echo -e "${BLUE}▶ Starting Sequential Version (Python/NumPy)${NC}"
    echo ""
    $PYTHON "$SRC_DIR/python/game_of_life_sequential.py" "$@"
}

run_cuda() {
    echo -e "${GREEN}▶ Starting CUDA Version${NC}"
    echo ""
    
    # Check if nvcc is available
    if ! command -v nvcc &> /dev/null; then
        echo -e "${RED}Error: CUDA toolkit not found. Please install CUDA.${NC}"
        exit 1
    fi
    
    # Create build directory if it doesn't exist
    mkdir -p "$BUILD_DIR"
    
    local CUDA_SRC="$SRC_DIR/cuda/game_of_life.cu"
    local CUDA_BIN="$BUILD_DIR/game_of_life"
    
    # Compile if needed
    if [ ! -f "$CUDA_BIN" ] || [ "$CUDA_SRC" -nt "$CUDA_BIN" ]; then
        echo -e "${YELLOW}Compiling CUDA code...${NC}"
        nvcc -o "$CUDA_BIN" "$CUDA_SRC"
        if [ $? -ne 0 ]; then
            echo -e "${RED}Compilation failed!${NC}"
            exit 1
        fi
        echo -e "${GREEN}Compilation successful!${NC}"
        echo ""
    fi
    
    "$CUDA_BIN" "$@"
}

run_benchmarks() {
    echo -e "${CYAN}▶ Running All Benchmarks${NC}"
    echo ""
    
    # Create benchmarks directory if it doesn't exist
    mkdir -p "$BENCHMARK_DIR"
    
    # Store current directory and change to benchmark dir for output files
    pushd "$BENCHMARK_DIR" > /dev/null
    
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}                  PYTHON SEQUENTIAL BENCHMARK                ${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    $PYTHON "$SRC_DIR/python/game_of_life_sequential.py" --benchmark
    echo ""
    
    if command -v nvcc &> /dev/null; then
        echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${GREEN}                     CUDA PARALLEL BENCHMARK                 ${NC}"
        echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        
        # Create build directory if it doesn't exist
        mkdir -p "$BUILD_DIR"
        
        local CUDA_SRC="$SRC_DIR/cuda/game_of_life.cu"
        local CUDA_BIN="$BUILD_DIR/game_of_life"
        
        # Compile if needed
        if [ ! -f "$CUDA_BIN" ] || [ "$CUDA_SRC" -nt "$CUDA_BIN" ]; then
            echo -e "${YELLOW}Compiling CUDA code...${NC}"
            nvcc -o "$CUDA_BIN" "$CUDA_SRC"
        fi
        
        "$CUDA_BIN" --benchmark
    else
        echo -e "${YELLOW}Skipping CUDA benchmark (nvcc not found)${NC}"
    fi
    
    popd > /dev/null
    
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}Results saved to: benchmarks/benchmark_*.csv${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

clean() {
    echo -e "${YELLOW}Cleaning build artifacts...${NC}"
    rm -rf "$BUILD_DIR"
    echo -e "${GREEN}Done!${NC}"
}

run_all() {
    echo -e "${CYAN}▶ Running All Implementations${NC}"
    echo ""
    
    # Default parameters
    local width=${1:-256}
    local height=${2:-256}
    local generations=${3:-100}
    
    echo -e "${CYAN}Parameters: ${width}x${height} grid, ${generations} generations${NC}"
    echo ""
    
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}                     PYTHON SEQUENTIAL                        ${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    $PYTHON "$SRC_DIR/python/game_of_life_sequential.py" "$width" "$height" "$generations"
    
    echo ""
    
    if command -v nvcc &> /dev/null; then
        echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${GREEN}                        CUDA PARALLEL                         ${NC}"
        echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        
        mkdir -p "$BUILD_DIR"
        local CUDA_SRC="$SRC_DIR/cuda/game_of_life.cu"
        local CUDA_BIN="$BUILD_DIR/game_of_life"
        
        if [ ! -f "$CUDA_BIN" ] || [ "$CUDA_SRC" -nt "$CUDA_BIN" ]; then
            echo -e "${YELLOW}Compiling CUDA code...${NC}"
            nvcc -o "$CUDA_BIN" "$CUDA_SRC"
            echo ""
        fi
        
        "$CUDA_BIN" "$width" "$height" "$generations"
    else
        echo -e "${YELLOW}Skipping CUDA (nvcc not found)${NC}"
    fi
}

# Main entry point
show_banner

if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

command=$1
shift  # Remove first argument, pass rest to subcommands

case $command in
    all|a)
        run_all "$@"
        ;;
    visual|v)
        run_visual "$@"
        ;;
    sequential|s)
        run_sequential "$@"
        ;;
    cuda|c)
        run_cuda "$@"
        ;;
    benchmark|b)
        run_benchmarks
        ;;
    clean)
        clean
        ;;
    help|h|--help|-h)
        show_help
        ;;
    *)
        echo -e "${RED}Unknown command: $command${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac
