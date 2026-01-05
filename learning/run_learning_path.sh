#!/bin/bash
# Interactive learning path runner

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         RLM Learning Path - Interactive Runner                 ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Activate environment
source ~/pyenv/rlm/bin/activate

cd "$(dirname "$0")"

phases=(
    "learn_01_basic.py:Phase 1: Basic Code Execution (5 min)"
    "learn_02_iterative.py:Phase 2: Iterative Reasoning (10 min)"
    "learn_03_recursive.py:Phase 3: Recursive Sub-Calls (15 min)"
    "learn_04_comparison.py:Phase 4: RLM vs Regular LLM (10 min)"
    "learn_05_visualize.py:Phase 5: Complex Trajectory for Visualization (15 min)"
)

echo -e "${GREEN}Available phases:${NC}"
echo ""
for i in "${!phases[@]}"; do
    IFS=':' read -r file desc <<< "${phases[$i]}"
    echo -e "  ${YELLOW}$((i+1)).${NC} $desc"
done
echo ""
echo -e "  ${YELLOW}A.${NC} Run all phases sequentially"
echo -e "  ${YELLOW}Q.${NC} Quit"
echo ""

while true; do
    read -p "Select phase (1-5, A, or Q): " choice
    
    case $choice in
        [1-5])
            idx=$((choice-1))
            IFS=':' read -r file desc <<< "${phases[$idx]}"
            echo ""
            echo -e "${BLUE}Running: $desc${NC}"
            echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
            python "$file"
            echo ""
            echo -e "${GREEN}✓ Phase complete!${NC}"
            echo ""
            read -p "Press Enter to continue..."
            echo ""
            ;;
        [Aa])
            for phase in "${phases[@]}"; do
                IFS=':' read -r file desc <<< "$phase"
                echo ""
                echo -e "${BLUE}Running: $desc${NC}"
                echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
                python "$file"
                echo ""
                echo -e "${GREEN}✓ Phase complete!${NC}"
                echo ""
                read -p "Press Enter for next phase..."
            done
            echo ""
            echo -e "${GREEN}🎉 All phases complete!${NC}"
            echo ""
            echo "Next steps:"
            echo "  1. cd visualizer/"
            echo "  2. npm install"
            echo "  3. npm run dev"
            echo "  4. Open http://localhost:3001"
            echo "  5. Upload .jsonl files from ./logs/"
            break
            ;;
        [Qq])
            echo "Goodbye!"
            exit 0
            ;;
        *)
            echo "Invalid choice. Please select 1-5, A, or Q."
            ;;
    esac
done

