import json
import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize W&B (offline mode)
wandb.init(project="llava-results", name="MultipleChoice_accuracy_analysis", mode="offline")

# Load results
results_file = '/data/llava/LLaVA/results/results.json'
with open(results_file, 'r') as f:
    results = json.load(f)

# Prepare DataFrame
df = pd.DataFrame(results)

# Check correctness explicitly
def is_correct(row):
    if (row['answer'] == "A" and row['ground_truth'] == "Left") or \
       (row['answer'] == "B" and row['ground_truth'] == "Right") or \
       (row['answer'] == "C" and row['ground_truth'] == "On top") or \
       (row['answer'] == "D" and row['ground_truth'] == "Under"):
        return True
    return False

df['correct'] = df.apply(is_correct, axis=1)

# Overall accuracy
accuracy = df['correct'].mean()
print(f"Overall Multiple Choice Accuracy: {accuracy * 100:.2f}%")

# Overall confidence (if available)
if 'confidence' in df.columns and df['confidence'].notna().any():
    overall_confidence = df['confidence'].mean()
    print(f"Overall Confidence: {overall_confidence * 100:.2f}%")
    wandb.log({"overall_confidence": overall_confidence})
else:
    print("Confidence data not available in results.")
    overall_confidence = None

print()
wandb.log({"overall_accuracy": accuracy})

# Accuracy per prompt type
prompt_accuracy = df.groupby('prompt_label')['correct'].mean().reset_index()
prompt_accuracy.columns = ['prompt_label', 'accuracy']

# Confidence per prompt type (if available)
if 'confidence' in df.columns and df['confidence'].notna().any():
    prompt_confidence = df.groupby('prompt_label')['confidence'].mean().reset_index()
    prompt_confidence.columns = ['prompt_label', 'confidence']
    prompt_stats = prompt_accuracy.merge(prompt_confidence, on='prompt_label')
else:
    prompt_stats = prompt_accuracy
    prompt_stats['confidence'] = None

# Print accuracy and confidence for each prompt
print("Accuracy & Confidence per Prompt Type:")
print("-" * 60)
for _, row in prompt_stats.iterrows():
    if row['confidence'] is not None:
        print(f"  {row['prompt_label']}: Acc={row['accuracy'] * 100:.2f}%, Conf={row['confidence'] * 100:.2f}%")
        wandb.log({f"accuracy_{row['prompt_label']}": row['accuracy']})
        wandb.log({f"confidence_{row['prompt_label']}": row['confidence']})
    else:
        print(f"  {row['prompt_label']}: Acc={row['accuracy'] * 100:.2f}%")
        wandb.log({f"accuracy_{row['prompt_label']}": row['accuracy']})
print("-" * 60)
print()

# Accuracy per ground truth (which option/direction was correct)
# Map ground truth to expected answer option
ground_truth_to_option = {
    'Left': 'A',
    'Right': 'B', 
    'On top': 'C',
    'Under': 'D'
}

# Add expected option column
df['expected_option'] = df['ground_truth'].map(ground_truth_to_option)

# Calculate accuracy per ground truth category
print("Accuracy per Ground Truth (Correct Answer Direction):")
print("-" * 60)
ground_truth_stats = df.groupby('ground_truth').agg({
    'correct': ['mean', 'sum', 'count']
}).reset_index()
ground_truth_stats.columns = ['ground_truth', 'accuracy', 'correct_count', 'total_count']

for _, row in ground_truth_stats.iterrows():
    option = ground_truth_to_option.get(row['ground_truth'], '?')
    print(f"  {row['ground_truth']:10} (Option {option}): {row['accuracy'] * 100:.2f}% ({int(row['correct_count'])}/{int(row['total_count'])} correct)")
    wandb.log({f"accuracy_ground_truth_{row['ground_truth']}": row['accuracy']})

print("-" * 60)
print()

# Plot accuracy per ground truth
plt.figure(figsize=(8,5))
colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']  # Green, Blue, Purple, Red
bars = plt.bar(ground_truth_stats['ground_truth'], ground_truth_stats['accuracy'], color=colors)
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.xlabel('Ground Truth (Correct Answer)')
plt.title('Accuracy per Ground Truth Direction')
# Add option labels on bars
for bar, gt in zip(bars, ground_truth_stats['ground_truth']):
    option = ground_truth_to_option.get(gt, '?')
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'Option {option}', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
wandb.log({"accuracy_per_ground_truth": wandb.Image(plt)})
plt.show()
plt.close()

# Plot accuracy per prompt type
plt.figure(figsize=(10,5))
sns.barplot(x='prompt_label', y='accuracy', data=prompt_stats, palette="viridis")
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.xlabel('Prompt Type')
plt.title('Accuracy per Prompt Type')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Log plot to W&B
wandb.log({"accuracy_per_prompt_type": wandb.Image(plt)})
plt.show()
plt.close()

# Plot confidence per prompt type (if available)
if 'confidence' in df.columns and df['confidence'].notna().any():
    plt.figure(figsize=(10,5))
    sns.barplot(x='prompt_label', y='confidence', data=prompt_stats, palette="magma")
    plt.ylabel('Confidence')
    plt.ylim(0, 1)
    plt.xlabel('Prompt Type')
    plt.title('Confidence per Prompt Type')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    wandb.log({"confidence_per_prompt_type": wandb.Image(plt)})
    plt.show()
    plt.close()
    
    # Plot accuracy vs confidence comparison
    fig, ax1 = plt.subplots(figsize=(12,6))
    
    x = range(len(prompt_stats))
    width = 0.35
    
    ax1.bar([i - width/2 for i in x], prompt_stats['accuracy'], width, label='Accuracy', color='steelblue')
    ax1.bar([i + width/2 for i in x], prompt_stats['confidence'], width, label='Confidence', color='coral')
    
    ax1.set_ylabel('Score')
    ax1.set_ylim(0, 1)
    ax1.set_xlabel('Prompt Type')
    ax1.set_title('Accuracy vs Confidence per Prompt Type')
    ax1.set_xticks(x)
    ax1.set_xticklabels(prompt_stats['prompt_label'], rotation=45, ha='right')
    ax1.legend()
    plt.tight_layout()
    
    wandb.log({"accuracy_vs_confidence": wandb.Image(plt)})
    plt.show()
    plt.close()

# Finish W&B logging
wandb.finish()