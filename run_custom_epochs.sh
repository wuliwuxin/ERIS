#!/bin/bash

# Custom training script with different epochs for different datasets
# Boiler: 20 epochs, Oppo: 20 epochs, Others: 50 epochs
# Using GPUs 0 and 1
# Results organized by dataset in separate folders

# Configuration
AVAILABLE_GPUS="0,1"
NUM_GPUS=2
BATCH_SIZE=${1:-64}  # Default batch size 64, can be overridden

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_color() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to print section headers
print_header() {
    local title=$1
    local length=${#title}
    local border=$(printf '=%.0s' $(seq 1 $((length + 20))))

    echo ""
    print_color $CYAN "$border"
    print_color $CYAN "$(printf '%*s' $(((${#border} + ${#title}) / 2)) "$title")"
    print_color $CYAN "$border"
    echo ""
}

# Dataset configurations with custom epochs
declare -A DATASET_CONFIG
DATASET_CONFIG[EMG]="100:0 1 2 3"         # 50 epochs, domains 0 1 2 3
DATASET_CONFIG[UCIHAR]="100:0 1 2 3 4"    # 50 epochs, domains 0 1 2 3 4
DATASET_CONFIG[Uni]="100:1 2 3 5"         # 50 epochs, domains 1 2 3 5
DATASET_CONFIG[Oppo]="100:S1 S2 S3 S4"    # 20 epochs, domains S1 S2 S3 S4
#DATASET_CONFIG[Boiler]="20:1 2 3"        # 20 epochs, domains 1 2 3

# Function to check system requirements
check_requirements() {
    print_header "SYSTEM REQUIREMENTS CHECK"

    # Check GPU availability
    if ! nvidia-smi >/dev/null 2>&1; then
        print_color $RED "❌ Error: nvidia-smi not found or GPUs not available"
        exit 1
    fi

    # Check specific GPUs
    for gpu in 0 1; do
        if ! nvidia-smi -i $gpu >/dev/null 2>&1; then
            print_color $RED "❌ Error: GPU $gpu not available"
            exit 1
        fi
    done

    print_color $GREEN "✓ GPUs 0 and 1 are available"

    # Check Python dependencies
    if ! python -c "import torch, numpy, sklearn, matplotlib" 2>/dev/null; then
        print_color $RED "❌ Error: Missing Python dependencies"
        exit 1
    fi

    print_color $GREEN "✓ Python dependencies available"

    # Check training scripts
    if [ ! -f "main.py" ]; then
        print_color $RED "❌ Error: main.py not found"
        exit 1
    fi

    print_color $GREEN "✓ Training scripts available"
}

# Function to display configuration
display_configuration() {
    print_header "CUSTOM TRAINING CONFIGURATION"

    echo "Hardware Configuration:"
    echo "  • Using GPUs: 0, 1"
    echo "  • Batch Size: $BATCH_SIZE"
    echo ""

    echo "Dataset Epoch Configuration:"
    for dataset in EMG UCIHAR Uni Oppo; do
        config=${DATASET_CONFIG[$dataset]}
        epochs=$(echo $config | cut -d':' -f1)
        domains=$(echo $config | cut -d':' -f2)
        domain_count=$(echo $domains | wc -w)
        echo "  • $dataset: $epochs epochs, $domain_count domains ($domains)"
    done
    echo ""

    # Calculate total training time estimation
    total_experiments=0
    total_gpu_hours=0
    for dataset in EMG UCIHAR Uni Oppo; do
        config=${DATASET_CONFIG[$dataset]}
        epochs=$(echo $config | cut -d':' -f1)
        domains=$(echo $config | cut -d':' -f2)
        domain_count=$(echo $domains | wc -w)
        total_experiments=$((total_experiments + domain_count))
        # Rough estimation: each epoch takes ~1 minute, domain takes epochs minutes
        domain_hours=$((epochs / 60))
        if [ $domain_hours -eq 0 ]; then domain_hours=1; fi
        total_gpu_hours=$((total_gpu_hours + domain_count * domain_hours))
    done

    echo "Estimation:"
    echo "  • Total Experiments: $total_experiments"
    echo "  • Estimated GPU-Hours: ~$total_gpu_hours hours"
    echo "  • Estimated Wall Time: ~$((total_gpu_hours / 2)) hours (with 2 GPUs)"
    echo ""
}

# Function to setup organized directories
setup_directories() {
    print_header "ORGANIZED DIRECTORY SETUP"

    # Create main directories
    mkdir -p ./results
    mkdir -p ./logs
    mkdir -p ./logs/master

    for dataset in EMG UCIHAR Uni Oppo; do
        # Results directories for each dataset
        mkdir -p "./results/$dataset"
        mkdir -p "./results/$dataset/models"
        mkdir -p "./results/$dataset/test_results"
        mkdir -p "./results/$dataset/visualizations"

        # Logs directories for each dataset
        mkdir -p "./logs/$dataset"

        # Set permissions
        chmod 755 "./results/$dataset"
        chmod 755 "./logs/$dataset"

        print_color $GREEN "✓ Created organized directories for $dataset"
    done

    # Create overall analysis directory
    mkdir -p "./results/analysis"
    chmod 755 ./results ./logs ./logs/master

    print_color $GREEN "✅ Organized directory structure created:"
    print_color $CYAN "📁 Results: ./results/[DATASET]/"
    print_color $CYAN "📝 Logs: ./logs/[DATASET]/"
    print_color $CYAN "📊 Analysis: ./results/analysis/"
}

# Function to train a single domain with organized output
train_domain() {
    local dataset=$1
    local domain=$2
    local epochs=$3
    local gpu_id=$4
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local log_file="./logs/${dataset}/train_${dataset}_domain_${domain}_gpu_${gpu_id}_${timestamp}.log"

    print_color $BLUE "🚀 Starting $dataset domain $domain training ($epochs epochs) on GPU $gpu_id..."

    # Use dataset-specific save directory
    local dataset_save_dir=$(realpath "./results/$dataset")

    # Run training with specific GPU  --seed $((42 + $gpu_id)) \
    CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
        --dataset $dataset \
        --target-domain $domain \
        --epochs $epochs \
        --batch-size $BATCH_SIZE \
        --gpu 0 \
        --mode train \
        --save-dir "$dataset_save_dir" \
        --seed 666666 \
        > $log_file 2>&1

    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        print_color $GREEN "✅ $dataset domain $domain completed successfully on GPU $gpu_id"

        # Wait a moment for file system to sync
        sleep 2

        # Check for model files in the models subdirectory (where they should be saved now)
        local model_pattern="$dataset_save_dir/models/Ours_${dataset}_target_domain_${domain}_*.pth"

        # Debug: Print what we're looking for and what exists
        print_color $CYAN "  Debug: Looking for pattern: $model_pattern"
        print_color $CYAN "  Debug: Files in models dir:"
        ls -la "$dataset_save_dir/models/" | grep "Ours_${dataset}_target_domain_${domain}" || true

        if ls $model_pattern 1> /dev/null 2>&1; then
            local model_file=$(ls $model_pattern | head -1)
            print_color $GREEN "  └─ Model saved: $dataset/models/$(basename $model_file)"
        else
            print_color $YELLOW "  └─ Warning: Model file not found with pattern $model_pattern"
            # List all files in models directory for debugging
            print_color $YELLOW "     Available files in models/:"
            ls "$dataset_save_dir/models/" 2>/dev/null | sed 's/^/       /' || print_color $YELLOW "     (directory empty or doesn't exist)"
        fi
    else
        print_color $RED "❌ $dataset domain $domain failed on GPU $gpu_id (exit code: $exit_code)"
        print_color $RED "   Check log: $log_file"

        # Show last few lines of log for debugging
        if [ -f "$log_file" ]; then
            print_color $YELLOW "   Last 5 lines of log:"
            tail -5 "$log_file" | sed 's/^/      /'
        fi
    fi

    return $exit_code
}

# Function to evaluate a domain with organized output
evaluate_domain() {
    local dataset=$1
    local domain=$2
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local log_file="./logs/${dataset}/eval_${dataset}_domain_${domain}_${timestamp}.log"

    print_color $YELLOW "📊 Evaluating $dataset domain $domain..."

    # Use dataset-specific save directory
    local dataset_save_dir=$(realpath "./results/$dataset")

    # Wait a bit to ensure file system sync
    sleep 1

    # Check for model file in models subdirectory with more flexible pattern matching
    print_color $CYAN "  Debug: Looking in directory: $dataset_save_dir/models/"

    # First, try the exact pattern
    local model_pattern="$dataset_save_dir/models/Ours_${dataset}_target_domain_${domain}_*.pth"
    local model_files=($(ls $model_pattern 2>/dev/null))

    # If no exact match, try case-insensitive or alternative patterns
    if [ ${#model_files[@]} -eq 0 ]; then
        # Try alternative patterns for debugging
        print_color $CYAN "  Debug: Trying alternative patterns..."
        model_files=($(find "$dataset_save_dir/models/" -name "*${dataset}*${domain}*.pth" 2>/dev/null))
    fi

    if [ ${#model_files[@]} -gt 0 ]; then
        local model_file="${model_files[0]}"  # Use the first match
        print_color $GREEN "  └─ Model found: $(basename $model_file)"

        # Run evaluation
        python main.py \
            --dataset $dataset \
            --target-domain $domain \
            --mode test \
            --save-dir "$dataset_save_dir" \
            --model-path "$model_file" \
            --seed 666666 \
            > $log_file 2>&1

        local exit_code=$?

        if [ $exit_code -eq 0 ]; then
            print_color $GREEN "✅ $dataset domain $domain evaluation completed"

            # Check for test results file in test_results subdirectory
            local results_file="$dataset_save_dir/test_results/Ours_${dataset}_test_results_target_domain_${domain}.json"
            if [ -f "$results_file" ]; then
                print_color $GREEN "  └─ Results saved: $dataset/test_results/$(basename $results_file)"
            else
                print_color $YELLOW "  └─ Warning: Test results file not found at $results_file"
            fi
        else
            print_color $YELLOW "⚠️  $dataset domain $domain evaluation failed (exit code: $exit_code)"
            print_color $YELLOW "   Check log: $log_file"

            # Show last few lines of log for debugging
            if [ -f "$log_file" ]; then
                print_color $YELLOW "   Last 5 lines of evaluation log:"
                tail -5 "$log_file" | sed 's/^/      /'
            fi
        fi
    else
        print_color $RED "❌ No model found for $dataset domain $domain"
        print_color $RED "   Pattern searched: $model_pattern"
        print_color $RED "   Available files in models/:"
        ls -la "$dataset_save_dir/models/" 2>/dev/null | sed 's/^/      /' || print_color $RED "      (directory empty or doesn't exist)"
        return 1
    fi

    return $exit_code
}

# Function to train all domains for a dataset using 2 GPUs
train_dataset_parallel() {
    local dataset=$1
    local config=${DATASET_CONFIG[$dataset]}
    local epochs=$(echo $config | cut -d':' -f1)
    local domains_str=$(echo $config | cut -d':' -f2)
    local domains=($domains_str)
    local num_domains=${#domains[@]}

    print_header "TRAINING DATASET: $dataset ($epochs epochs)"

    local pids=()
    local gpu_assignments=()

    # Launch training jobs in parallel using both GPUs
    for i in "${!domains[@]}"; do
        local domain=${domains[$i]}
        local gpu_id=$((i % 2))  # Alternate between GPU 0 and 1

        print_color $BLUE "Launching $dataset domain $domain on GPU $gpu_id..."
        train_domain $dataset $domain $epochs $gpu_id &
        pids+=($!)
        gpu_assignments+=("$dataset:$domain:GPU$gpu_id")

        # Small delay to prevent startup conflicts
        sleep 3
    done

    print_color $CYAN "All $dataset training jobs launched. Waiting for completion..."
    print_color $CYAN "Active jobs: ${#pids[@]}"

    # Wait for all training jobs to complete
    local failed_jobs=0
    for i in "${!pids[@]}"; do
        local pid=${pids[$i]}
        local assignment=${gpu_assignments[$i]}

        print_color $YELLOW "Waiting for $assignment (PID: $pid)..."
        wait $pid
        local exit_code=$?

        if [ $exit_code -ne 0 ]; then
            print_color $RED "⚠️  Training failed for $assignment"
            failed_jobs=$((failed_jobs + 1))
        fi
    done

    print_color $CYAN "$dataset training completed. Failed jobs: $failed_jobs/$num_domains"

    # Show organized structure after training
    show_organized_structure $dataset

    # Wait a bit for file system to fully sync before evaluation
    print_color $PURPLE "Waiting for file system sync before evaluation..."
    sleep 5

    # Run evaluations sequentially
    print_color $PURPLE "Starting evaluations for $dataset..."
    local eval_failed=0
    for domain in "${domains[@]}"; do
        evaluate_domain $dataset $domain
        if [ $? -ne 0 ]; then
            eval_failed=$((eval_failed + 1))
        fi
        sleep 2  # Small delay between evaluations
    done

    print_color $CYAN "$dataset evaluation completed. Failed evaluations: $eval_failed/$num_domains"

    # Show final organized structure
    show_organized_structure $dataset

    return $failed_jobs
}

# Function to show current organized structure
show_organized_structure() {
    local dataset=$1
    local dataset_dir="./results/$dataset"

    if [ -d "$dataset_dir" ]; then
        print_color $CYAN "\n📁 $dataset directory structure:"

        # Count files in each subdirectory
        local models_count=$(find "$dataset_dir/models" -name "*.pth" 2>/dev/null | wc -l)
        local tests_count=$(find "$dataset_dir/test_results" -name "*.json" 2>/dev/null | wc -l)
        local viz_count=$(find "$dataset_dir/visualizations" -name "*" -type f 2>/dev/null | wc -l)

        echo "  $dataset/"
        echo "  ├── models/ ($models_count .pth files)"
        echo "  ├── test_results/ ($tests_count .json files)"
        echo "  ├── visualizations/ ($viz_count files)"
        echo "  └── summary files"

        # Show recent files
        if [ $models_count -gt 0 ]; then
            print_color $GREEN "  Recent models:"
            find "$dataset_dir/models" -name "*.pth" -printf "%f\n" 2>/dev/null | tail -3 | sed 's/^/    /'
        fi

        if [ $tests_count -gt 0 ]; then
            print_color $GREEN "  Recent test results:"
            find "$dataset_dir/test_results" -name "*.json" -printf "%f\n" 2>/dev/null | tail -3 | sed 's/^/    /'
        fi
    fi
}

# Function to create dataset summary with organized structure
create_dataset_summary() {
    local dataset=$1
    local dataset_dir=$(realpath "./results/$dataset")

    print_color $CYAN "📊 Creating summary for $dataset..."

    python -c "
import os
import json
import glob
import pandas as pd
import numpy as np

dataset = '$dataset'
dataset_dir = '$dataset_dir'

print(f'Analyzing {dataset} in: {dataset_dir}')

# Look for test results in the organized structure
test_results_dir = os.path.join(dataset_dir, 'test_results')
result_files = glob.glob(os.path.join(test_results_dir, '*.json'))

print(f'Found {len(result_files)} test result files in {test_results_dir}')

results = []
for result_file in result_files:
    try:
        print(f'Processing: {os.path.basename(result_file)}')
        with open(result_file, 'r') as f:
            data = json.load(f)

        # Extract domain from filename
        filename = os.path.basename(result_file)
        if 'target_domain_' in filename:
            domain = filename.split('target_domain_')[1].split('.json')[0]
        else:
            domain = 'unknown'

        result = {
            'domain': domain,
            'accuracy': data.get('accuracy', 0),
            'ece': data.get('ece', 0),
            'f1_micro': data.get('f1_micro', 0),
            'f1_macro': data.get('f1_macro', 0),
            'f1_weighted': data.get('f1_weighted', 0),
            'precision_macro': data.get('precision_macro', 0),
            'precision_weighted': data.get('precision_weighted', 0),
            'recall_macro': data.get('recall_macro', 0),
            'recall_weighted': data.get('recall_weighted', 0)
        }
        results.append(result)
        print(f'  Success: domain {domain}, accuracy {result[\"accuracy\"]:.4f}')

    except Exception as e:
        print(f'Error processing {result_file}: {e}')

if results:
    df = pd.DataFrame(results)

    print(f'\\n{dataset} Results Summary:')
    print('=' * 70)
    print(df.round(4).to_string(index=False))

    # Calculate statistics
    numeric_cols = ['accuracy', 'ece', 'f1_micro', 'f1_macro', 'f1_weighted']
    available_cols = [col for col in numeric_cols if col in df.columns]

    if available_cols:
        stats = df[available_cols].agg(['mean', 'std', 'min', 'max'])
        print(f'\\n{dataset} Statistics:')
        print('=' * 50)
        print(stats.round(4).to_string())

    # Save results in dataset directory
    summary_file = os.path.join(dataset_dir, f'{dataset}_summary.csv')
    df.to_csv(summary_file, index=False)

    if available_cols:
        stats_file = os.path.join(dataset_dir, f'{dataset}_statistics.csv')
        stats.to_csv(stats_file)

    # Best performing domain
    if 'accuracy' in df.columns:
        best_acc_idx = df['accuracy'].idxmax()
        best_domain = df.loc[best_acc_idx]
        print(f'\\nBest performing domain: {best_domain[\"domain\"]} (Accuracy: {best_domain[\"accuracy\"]:.4f})')

    print(f'\\nFiles saved:')
    print(f'  • {summary_file}')
    if available_cols:
        print(f'  • {stats_file}')

    # Count model files
    models_dir = os.path.join(dataset_dir, 'models')
    model_count = len(glob.glob(os.path.join(models_dir, '*.pth')))
    print(f'  • Model files in models/: {model_count}')

else:
    print(f'No valid results found for {dataset}')
"
}

# Function to create comprehensive final report with organized structure
create_final_report() {
    print_header "CREATING COMPREHENSIVE REPORT"

    local analysis_dir="./results/analysis"
    mkdir -p "$analysis_dir"

    # Create a Python script for the report to avoid shell escaping issues
    cat > /tmp/create_report.py << 'EOF'
import os
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys

# Set style for plots
plt.style.use('default')
sns.set_palette('husl')

#datasets = ['EMG', 'UCIHAR', 'Uni', 'Oppo', 'Boiler']
#dataset_epochs = {'EMG': 50, 'UCIHAR': 50, 'Uni': 50, 'Oppo': 20, 'Boiler': 20}
datasets = ['EMG', 'UCIHAR', 'Uni', 'Oppo']
dataset_epochs = {'EMG': 100, 'UCIHAR': 100, 'Uni': 100, 'Oppo': 50}
results_base_dir = './results'
analysis_dir = sys.argv[1] if len(sys.argv) > 1 else './results/analysis'

all_results = []
dataset_summaries = {}

print(f'Collecting results from organized structure: {results_base_dir}')

for dataset in datasets:
    print(f'\nProcessing {dataset}...')

    # Look in dataset-specific test_results directory
    dataset_dir = os.path.join(results_base_dir, dataset)
    test_results_dir = os.path.join(dataset_dir, 'test_results')

    result_files = []
    if os.path.exists(test_results_dir):
        result_files = glob.glob(os.path.join(test_results_dir, '*.json'))

    print(f'  Found {len(result_files)} result files')

    dataset_results = []
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)

            filename = os.path.basename(result_file)
            if 'target_domain_' in filename:
                domain = filename.split('target_domain_')[1].split('.json')[0]
            else:
                domain = 'unknown'

            result = {
                'dataset': dataset,
                'domain': domain,
                'epochs': dataset_epochs[dataset],
                'accuracy': data.get('accuracy', 0),
                'ece': data.get('ece', 0),
                'f1_micro': data.get('f1_micro', 0),
                'f1_macro': data.get('f1_macro', 0),
                'f1_weighted': data.get('f1_weighted', 0),
                'precision_macro': data.get('precision_macro', 0),
                'precision_weighted': data.get('precision_weighted', 0),
                'recall_macro': data.get('recall_macro', 0),
                'recall_weighted': data.get('recall_weighted', 0)
            }

            dataset_results.append(result)
            all_results.append(result)
            print(f'    {domain}: {result["accuracy"]:.4f} accuracy')

        except Exception as e:
            print(f'Error processing {result_file}: {e}')

    if dataset_results:
        df = pd.DataFrame(dataset_results)

        summary = {
            'dataset': dataset,
            'epochs': dataset_epochs[dataset],
            'num_experiments': len(dataset_results),
            'avg_accuracy': df['accuracy'].mean(),
            'std_accuracy': df['accuracy'].std(),
            'avg_ece': df['ece'].mean(),
            'std_ece': df['ece'].std(),
            'avg_f1_macro': df['f1_macro'].mean(),
            'best_accuracy': df['accuracy'].max(),
            'worst_accuracy': df['accuracy'].min()
        }

        dataset_summaries[dataset] = summary
        print(f'  Summary: {summary["avg_accuracy"]:.4f} avg accuracy')

if all_results:
    print(f'\nTotal results collected: {len(all_results)}')

    # Create comprehensive DataFrame
    all_df = pd.DataFrame(all_results)

    # Save detailed results
    detailed_file = os.path.join(analysis_dir, 'all_detailed_results.csv')
    all_df.to_csv(detailed_file, index=False)
    print(f'Saved detailed results: {detailed_file}')

    # Create summary DataFrame
    if dataset_summaries:
        summary_df = pd.DataFrame(list(dataset_summaries.values()))
        summary_file = os.path.join(analysis_dir, 'overall_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        print(f'Saved summary: {summary_file}')

        # Create visualization
        if len(summary_df) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('ERIS Epochs Training Results', fontsize=16, fontweight='bold')

            # 1. Accuracy by dataset
            datasets_with_data = summary_df['dataset'].tolist()
            accuracy_data = []
            ece_data = []

            for dataset in datasets_with_data:
                dataset_results = [r['accuracy'] for r in all_results if r['dataset'] == dataset]
                accuracy_data.append(dataset_results)

                dataset_ece = [r['ece'] for r in all_results if r['dataset'] == dataset]
                ece_data.append(dataset_ece)

            if accuracy_data:
                bp1 = axes[0, 0].boxplot(accuracy_data, labels=datasets_with_data, patch_artist=True)
                for patch, dataset in zip(bp1['boxes'], datasets_with_data):
                    if dataset_epochs[dataset] == 20:
                        patch.set_facecolor('lightcoral')
                    else:
                        patch.set_facecolor('lightblue')
                axes[0, 0].set_title('Accuracy Distribution by Dataset', fontweight='bold')
                axes[0, 0].set_ylabel('Accuracy')
                axes[0, 0].grid(True, alpha=0.3)
                axes[0, 0].tick_params(axis='x', rotation=45)

            # 2. ECE by dataset
            if ece_data:
                bp2 = axes[0, 1].boxplot(ece_data, labels=datasets_with_data, patch_artist=True)
                for patch, dataset in zip(bp2['boxes'], datasets_with_data):
                    if dataset_epochs[dataset] == 20:
                        patch.set_facecolor('lightcoral')
                    else:
                        patch.set_facecolor('lightblue')
                axes[0, 1].set_title('ECE Distribution by Dataset', fontweight='bold')
                axes[0, 1].set_ylabel('ECE')
                axes[0, 1].grid(True, alpha=0.3)
                axes[0, 1].tick_params(axis='x', rotation=45)

            # 3. Epochs vs Average Accuracy
            if not summary_df.empty:
                for _, row in summary_df.iterrows():
                    color = 'red' if row['epochs'] == 20 else 'blue'
                    axes[1, 0].scatter(row['epochs'], row['avg_accuracy'],
                                     label=row['dataset'], s=150, alpha=0.7, color=color)
                    axes[1, 0].annotate(row['dataset'],
                                      (row['epochs'], row['avg_accuracy']),
                                      xytext=(5, 5), textcoords='offset points')

                axes[1, 0].set_xlabel('Training Epochs')
                axes[1, 0].set_ylabel('Average Accuracy')
                axes[1, 0].set_title('Training Epochs vs Average Accuracy', fontweight='bold')
                axes[1, 0].grid(True, alpha=0.3)

            # 4. Performance comparison
            if not summary_df.empty:
                x_pos = np.arange(len(summary_df))
                width = 0.35

                bars1 = axes[1, 1].bar(x_pos - width/2, summary_df['avg_accuracy'],
                                      width, label='Accuracy', alpha=0.7)
                bars2 = axes[1, 1].bar(x_pos + width/2, summary_df['avg_f1_macro'],
                                      width, label='F1 Macro', alpha=0.7)

                axes[1, 1].set_xlabel('Dataset')
                axes[1, 1].set_ylabel('Score')
                axes[1, 1].set_title('Accuracy vs F1 Macro by Dataset', fontweight='bold')
                axes[1, 1].set_xticks(x_pos)
                axes[1, 1].set_xticklabels(summary_df['dataset'])
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            plot_file = os.path.join(analysis_dir, 'comprehensive_analysis.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f'Saved visualization: {plot_file}')

        # Create detailed report
        report_file = os.path.join(analysis_dir, 'comprehensive_report.md')
        with open(report_file, 'w') as f:
            f.write('# ERIS Epochs Training Report\n\n')
            f.write(f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')

            f.write('## Directory Structure\n\n')
            f.write('Results are organized by dataset:\n\n')
            f.write('```\n')
            f.write('results/\n')
            for dataset in datasets:
                if dataset in dataset_summaries:
                    f.write(f'├── {dataset}/\n')
                    f.write(f'│   ├── models/           # .pth files\n')
                    f.write(f'│   ├── test_results/     # .json files\n')
                    f.write(f'│   ├── visualizations/   # plots\n')
                    f.write(f'│   ├── {dataset}_summary.csv\n')
                    f.write(f'│   └── {dataset}_statistics.csv\n')
            f.write('└── analysis/            # Overall analysis\n')
            f.write('```\n\n')

            f.write('## Training Configuration\n\n')
            f.write('| Dataset | Epochs | Domains |\n')
            f.write('|---------|--------|---------|\n')
            for dataset in datasets:
                if dataset in dataset_summaries:
                    epochs = dataset_epochs[dataset]
                    num_exp = dataset_summaries[dataset]['num_experiments']
                    f.write(f'| {dataset} | {epochs} | {num_exp} |\n')
            f.write('\n')

            if dataset_summaries:
                f.write('## Executive Summary\n\n')
                total_experiments = sum(s['num_experiments'] for s in dataset_summaries.values())
                overall_avg_acc = np.mean([s['avg_accuracy'] for s in dataset_summaries.values()])
                overall_avg_ece = np.mean([s['avg_ece'] for s in dataset_summaries.values()])

                short_epoch_datasets = [d for d, s in dataset_summaries.items() if dataset_epochs[d] == 20]
                long_epoch_datasets = [d for d, s in dataset_summaries.items() if dataset_epochs[d] == 50]

                f.write(f'- **Total Experiments**: {total_experiments}\n')
                f.write(f'- **Overall Average Accuracy**: {overall_avg_acc:.4f}\n')
                f.write(f'- **Overall Average ECE**: {overall_avg_ece:.4f}\n')
                f.write(f'- **Datasets with 20 epochs**: {len(short_epoch_datasets)} ({", ".join(short_epoch_datasets)})\n')
                f.write(f'- **Datasets with 50 epochs**: {len(long_epoch_datasets)} ({", ".join(long_epoch_datasets)})\n')

                if short_epoch_datasets and long_epoch_datasets:
                    short_avg_acc = np.mean([dataset_summaries[d]['avg_accuracy'] for d in short_epoch_datasets])
                    long_avg_acc = np.mean([dataset_summaries[d]['avg_accuracy'] for d in long_epoch_datasets])
                    f.write(f'- **20-epoch datasets avg accuracy**: {short_avg_acc:.4f}\n')
                    f.write(f'- **50-epoch datasets avg accuracy**: {long_avg_acc:.4f}\n')
                f.write('\n')

            if dataset_summaries:
                f.write('## Dataset Performance Summary\n\n')
                summary_df = pd.DataFrame(list(dataset_summaries.values()))
                f.write(summary_df.round(4).to_markdown(index=False))
                f.write('\n\n')

                f.write('## Best Performers\n\n')
                best_acc_dataset = summary_df.loc[summary_df['avg_accuracy'].idxmax()]
                best_ece_dataset = summary_df.loc[summary_df['avg_ece'].idxmin()]

                f.write(f'- **Highest Average Accuracy**: {best_acc_dataset["dataset"]} ({best_acc_dataset["avg_accuracy"]:.4f} with {best_acc_dataset["epochs"]} epochs)\n')
                f.write(f'- **Best Calibration (Lowest ECE)**: {best_ece_dataset["dataset"]} ({best_ece_dataset["avg_ece"]:.4f} with {best_ece_dataset["epochs"]} epochs)\n\n')

            f.write('## Hardware Configuration\n\n')
            f.write('- **GPUs Used**: 0, 1\n')
            f.write('- **Parallel Training**: Domains within each dataset trained in parallel\n\n')

            f.write('## Files Generated\n\n')
            f.write('- **Individual Dataset Folders**: Each containing organized models, test results, and summaries\n')
            f.write('- **Analysis Folder**: Contains overall analysis and comparisons\n')
            f.write('  - `all_detailed_results.csv`: Complete experimental data\n')
            f.write('  - `overall_summary.csv`: Dataset-level summary statistics\n')
            f.write('  - `comprehensive_analysis.png`: Visualization plots\n')

        print(f'Saved report: {report_file}')
        print(f'\n✅ Comprehensive analysis completed!')
        print(f'Analysis files location: {analysis_dir}')

else:
    print(f'\n❌ No results found for analysis')
EOF

    # Run the Python script
    python /tmp/create_report.py "$analysis_dir"

    # Clean up the temporary script
    rm -f /tmp/create_report.py
}

# Function to monitor GPU usage
monitor_gpu_usage() {
    local log_file="./logs/master/gpu_monitoring.log"

    while true; do
        echo "$(date): GPU Status" >> $log_file
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader | head -2 >> $log_file
        echo "" >> $log_file
        sleep 60  # Log every minute
    done &

    echo $! > ./logs/master/monitor_pid.txt
    print_color $GREEN "✓ GPU monitoring started (PID: $(cat ./logs/master/monitor_pid.txt))"
}

# Function to stop GPU monitoring
stop_gpu_monitoring() {
    if [ -f "./logs/master/monitor_pid.txt" ]; then
        local monitor_pid=$(cat ./logs/master/monitor_pid.txt)
        if kill -0 $monitor_pid 2>/dev/null; then
            kill $monitor_pid
            print_color $GREEN "✓ GPU monitoring stopped"
        fi
        rm -f ./logs/master/monitor_pid.txt
    fi
}

# Main execution function
main() {
    local start_time=$(date +%s)

    # Create master log
    exec > >(tee -a "./logs/master/custom_training_$(date +%Y%m%d_%H%M%S).log")
    exec 2>&1

    print_header "ERIS EPOCHS TRAINING PIPELINE"

    print_color $CYAN "🚀 Starting ERIS epochs training pipeline..."
    print_color $CYAN "📅 Started at: $(date)"
    print_color $CYAN "📁 Results will be organized by dataset"

    # System checks and setup
    check_requirements
    display_configuration
    setup_directories

    # Start monitoring
    monitor_gpu_usage

    # Train each dataset
    local total_failed=0
#    local dataset_order=(Boiler Oppo EMG UCIHAR Uni)  # Start with shorter ones
    local dataset_order=(EMG UCIHAR Uni Oppo)  # Start with shorter ones
    for dataset in "${dataset_order[@]}"; do
        print_header "PROCESSING DATASET: $dataset"

        train_dataset_parallel $dataset
        local failed=$?
        total_failed=$((total_failed + failed))

        # Create individual dataset summary
        create_dataset_summary $dataset

        print_color $PURPLE "Completed $dataset. Moving to next dataset..."
        sleep 5  # Brief pause between datasets
    done

    # Stop monitoring
    stop_gpu_monitoring

    # Create final comprehensive report
    create_final_report

    # Final summary
    local end_time=$(date +%s)
    local total_duration=$((end_time - start_time))

    print_header "TRAINING COMPLETED"

    print_color $GREEN "🎉 ERIS epochs training pipeline completed!"
    print_color $CYAN "📊 Total execution time: $((total_duration / 3600)) hours $((total_duration % 3600 / 60)) minutes"
    print_color $CYAN "⚠️  Total failed jobs across all datasets: $total_failed"

    # Show final organized structure
    print_header "FINAL ORGANIZED STRUCTURE"

#    for dataset in EMG UCIHAR Uni Oppo Boiler; do
#        show_organized_structure $dataset
#    done
    for dataset in EMG UCIHAR Uni Oppo; do
        show_organized_structure $dataset
    done

    print_color $CYAN "\n📁 Main directories:"
    print_color $CYAN "  • Results: $(realpath ./results)/"
    print_color $CYAN "  • Analysis: $(realpath ./results/analysis)/"
    print_color $CYAN "  • Logs: $(realpath ./logs)/"

    # Final GPU status
    print_color $BLUE "\n🔧 Final GPU status:"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader | head -2

    print_color $GREEN "\n✅ All results are organized by dataset!"
    print_color $GREEN "✅ Check individual dataset folders for detailed results!"
}

# Cleanup function
cleanup() {
    print_color $YELLOW "🧹 Cleaning up..."
    stop_gpu_monitoring

    # Kill any remaining background processes
    jobs -p | xargs -r kill 2>/dev/null

    print_color $GREEN "✓ Cleanup completed"
}

trap cleanup EXIT

# Script header
print_color $PURPLE "
╔═══════════════════════════════════════════════════════════════════════════════╗
║                  ERIS EPOCHS TRAINING PIPELINE                                ║
║           Oppo: 50, EMG: 100, UCIHAR: 100, Uni: 100 epochs           ║
║                     Using GPUs 0 & 1 - Results Organized by Dataset           ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"

# Execute main function
main "$@"