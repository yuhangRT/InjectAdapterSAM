#!/bin/bash
# Grid search script for SAM+CRNet adapter experiments.
# Tests different adapter sizes and compression ratios.

# Configuration
ADAPTER_SIZES=("small" "medium" "large")
COMPRESSION_RATIOS=(4 8 16 32 64)
PROMPT_STRATEGIES=("random" "center" "box")

# Training configuration
DATASET="coco"
DATA_DIR="/path/to/coco"  # Update this path
SAM_CHECKPOINT="./checkpoints/sam_vit_h_4b8939.pth"  # Update this path
EPOCHS=100
BATCH_SIZE=8
NUM_WORKERS=4
GPU=0

# Create logs directory
LOG_DIR="./logs/grid_search"
mkdir -p "$LOG_DIR"

# Create results file
RESULTS_FILE="$LOG_DIR/results.csv"
echo "adapter_size,compression_ratio,prompt_strategy,epoch,train_loss,val_iou,val_dice,test_iou,test_dice,training_time" > "$RESULTS_FILE"

# Count total experiments
TOTAL_EXPERIMENTS=${#ADAPTER_SIZES[@]} * ${#COMPRESSION_RATIOS[@]} * ${#PROMPT_STRATEGIES[@]}
CURRENT=0

echo "========================================"
echo "SAM+CRNet Adapter Grid Search"
echo "========================================"
echo "Total experiments: $TOTAL_EXPERIMENTS"
echo "Log directory: $LOG_DIR"
echo "Results file: $RESULTS_FILE"
echo "========================================"
echo ""

# Run experiments
for SIZE in "${ADAPTER_SIZES[@]}"; do
    for RATIO in "${COMPRESSION_RATIOS[@]}"; do
        for STRATEGY in "${PROMPT_STRATEGIES[@]}"; do
            CURRENT=$((CURRENT + 1))

            # Experiment name
            EXP_NAME="sam_${DATASET}_${SIZE}_cr${RATIO}_${STRATEGY}"
            LOG_FILE="$LOG_DIR/${EXP_NAME}.log"

            echo "========================================"
            echo "Experiment $CURRENT/$TOTAL_EXPERIMENTS: $EXP_NAME"
            echo "========================================"
            echo "Adapter size: $SIZE"
            echo "Compression ratio: 1/$RATIO"
            echo "Prompt strategy: $STRATEGY"
            echo "Log file: $LOG_FILE"
            echo ""

            # Create checkpoint directory
            CHECKPOINT_DIR="./checkpoints/grid_search/${EXP_NAME}"
            mkdir -p "$CHECKPOINT_DIR"

            # Start time
            START_TIME=$(date +%s)

            # Run training
            python main_sam.py \
                --mode sam \
                --sam-model-type vit_h \
                --sam-checkpoint "$SAM_CHECKPOINT" \
                --dataset "$DATASET" \
                --data-dir "$DATA_DIR" \
                --adapter-size "$SIZE" \
                --compression-ratio "$RATIO" \
                --prompt-strategy "$STRATEGY" \
                --num-prompts 1 \
                --epochs "$EPOCHS" \
                --batch-size "$BATCH_SIZE" \
                --workers "$NUM_WORKERS" \
                --gpu "$GPU" \
                --scheduler cosine \
                2>&1 | tee "$LOG_FILE"

            # End time
            END_TIME=$(date +%s)
            TRAINING_TIME=$((END_TIME - START_TIME))

            # Extract results from log
            # This is a simple parser - you may need to adjust based on your log format
            TEST_IOU=$(grep "Mean IoU:" "$LOG_FILE" | tail -1 | awk '{print $NF}')
            TEST_DICE=$(grep "Mean Dice:" "$LOG_FILE" | tail -1 | awk '{print $NF}')

            # Save results
            echo "$SIZE,$RATIO,$STRATEGY,$EPOCHS,-,-,-,$TEST_IOU,$TEST_DICE,$TRAINING_TIME" >> "$RESULTS_FILE"

            echo ""
            echo "Experiment $EXP_NAME completed in ${TRAINING_TIME}s"
            echo "Test IoU: $TEST_IOU"
            echo "Test Dice: $TEST_DICE"
            echo ""
        done
    done
done

echo "========================================"
echo "Grid search completed!"
echo "========================================"
echo "Results saved to: $RESULTS_FILE"
echo ""

# Display summary
echo "Results Summary:"
echo "================"
cat "$RESULTS_FILE"
echo ""

# Optional: Plot results if matplotlib is available
python3 << 'EOF'
import sys
import os
try:
    import pandas as pd
    import matplotlib.pyplot as plt

    results_file = sys.argv[1] if len(sys.argv) > 1 else "./logs/grid_search/results.csv"

    if os.path.exists(results_file):
        df = pd.read_csv(results_file)

        # Create summary plots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot 1: IoU vs compression ratio for each adapter size
        for size in ['small', 'medium', 'large']:
            data = df[df['adapter_size'] == size]
            if len(data) > 0 and 'test_iou' in data.columns:
                axes[0].plot(data['compression_ratio'], data['test_iou'], marker='o', label=size)
        axes[0].set_xlabel('Compression Ratio (1/x)')
        axes[0].set_ylabel('Test IoU')
        axes[0].set_title('IoU vs Compression Ratio')
        axes[0].legend()
        axes[0].grid(True)

        # Plot 2: IoU by adapter size
        if 'adapter_size' in df.columns and 'test_iou' in df.columns:
            df.boxplot(column='test_iou', by='adapter_size', ax=axes[1])
        axes[1].set_xlabel('Adapter Size')
        axes[1].set_ylabel('Test IoU')
        axes[1].set_title('IoU Distribution by Adapter Size')

        # Plot 3: Training time
        for size in ['small', 'medium', 'large']:
            data = df[df['adapter_size'] == size]
            if len(data) > 0 and 'training_time' in data.columns:
                axes[2].plot(data['compression_ratio'], data['training_time'] / 60, marker='o', label=size)
        axes[2].set_xlabel('Compression Ratio (1/x)')
        axes[2].set_ylabel('Training Time (minutes)')
        axes[2].set_title('Training Time vs Compression Ratio')
        axes[2].legend()
        axes[2].grid(True)

        plt.tight_layout()
        plt.savefig('./logs/grid_search/summary.png', dpi=150)
        print("\nSummary plot saved to: ./logs/grid_search/summary.png")

except ImportError:
    print("\nNote: Install pandas and matplotlib to generate summary plots")
    print("  pip install pandas matplotlib")
except Exception as e:
    print(f"\nNote: Could not generate plots: {e}")
EOF

python3 -c "import sys; sys.argv.append('$RESULTS_FILE')" << 'EOF'
# (The plotting code is above - this is just a placeholder)
pass
EOF
