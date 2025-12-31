import json
import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Initialize W&B (offline mode)
wandb.init(project="llava-results", name="MultipleChoice_accuracy_analysis", mode="offline")

# Load results
results_file = '/data/llava/LLaVA/results/results.json'
with open(results_file, 'r') as f:
    results = json.load(f)

# Prepare DataFrame
df = pd.DataFrame(results)

def extract_answer(answer):
    """
    Extract a single letter answer (A, B, C, or D) from various response formats.
    Returns the normalized letter or 'INVALID' if no clear answer can be determined.
    """
    if not answer or not isinstance(answer, str):
        return 'INVALID'
    
    answer = answer.strip()
    
    # Case 1: Single letter (with optional period)
    if re.match(r'^[A-D]\.?$', answer):
        return answer[0]
    
    # Case 2: Check if the answer ENDS with a letter (possibly with period)
    # This handles cases like "Based on my analysis... A" or "The answer is B."
    end_match = re.search(r'[A-D]\.?$', answer)
    if end_match:
        return end_match.group()[0]
    
    # Case 3: Model repeated the mapping without answering
    # e.g., "A = Left, B = Right, C = On top, D = Under" or "A, B, C, D"
    if re.match(r'^A\s*=\s*Left', answer) or answer == "A, B, C, D":
        return 'INVALID'
    
    # Case 4: Check for variations like "A = Left, B = Right, C = Table, D = Floor"
    # These are also invalid (model confused the mapping)
    if re.match(r'^A\s*=\s*\w+,\s*B\s*=', answer):
        return 'INVALID'
    
    # Case 5: Long descriptive answer - try to extract spatial relationship
    # Look for explicit statements about position
    answer_lower = answer.lower()
    
    # Check for clear positional statements
    # Priority: look for "is on the left", "to the left of", "left side", etc.
    left_patterns = [
        r'\bon the left\b', r'\bto the left\b', r'\bleft side\b', r'\bleft of\b',
        r'\bpositioned.*left\b', r'\bplaced.*left\b', r'\blocated.*left\b'
    ]
    right_patterns = [
        r'\bon the right\b', r'\bto the right\b', r'\bright side\b', r'\bright of\b',
        r'\bpositioned.*right\b', r'\bplaced.*right\b', r'\blocated.*right\b'
    ]
    top_patterns = [
        r'\bon top\b', r'\bon the top\b', r'\babove\b', r'\bresting on\b',
        r'\bsitting on\b', r'\bplaced on\b', r'\bon the chair\b', r'\bon the table\b',
        r'\bon the armchair\b', r'\bon the seat\b'
    ]
    under_patterns = [
        r'\bunder\b', r'\bunderneath\b', r'\bbeneath\b', r'\bbelow\b'
    ]
    
    # Count matches for each direction
    left_count = sum(1 for p in left_patterns if re.search(p, answer_lower))
    right_count = sum(1 for p in right_patterns if re.search(p, answer_lower))
    top_count = sum(1 for p in top_patterns if re.search(p, answer_lower))
    under_count = sum(1 for p in under_patterns if re.search(p, answer_lower))
    
    # If one direction clearly dominates, use it
    counts = {'A': left_count, 'B': right_count, 'C': top_count, 'D': under_count}
    max_count = max(counts.values())
    
    if max_count > 0:
        # Check if there's a clear winner (no tie)
        winners = [k for k, v in counts.items() if v == max_count]
        if len(winners) == 1:
            return winners[0]
    
    # If we can't determine, mark as invalid
    return 'INVALID'

# Apply answer extraction
df['extracted_answer'] = df['answer'].apply(extract_answer)

# Count invalid answers
invalid_count = (df['extracted_answer'] == 'INVALID').sum()
total_count = len(df)
print(f"Answer Extraction Summary:")
print(f"  Total responses: {total_count}")
print(f"  Valid answers: {total_count - invalid_count} ({(total_count - invalid_count) / total_count * 100:.1f}%)")
print(f"  Invalid/unclear answers: {invalid_count} ({invalid_count / total_count * 100:.1f}%)")
print()

# Show breakdown of invalid answers by prompt type
invalid_by_prompt = df[df['extracted_answer'] == 'INVALID'].groupby('prompt_label').size()
print("Invalid answers by prompt type:")
for prompt_type, count in invalid_by_prompt.items():
    total_for_type = len(df[df['prompt_label'] == prompt_type])
    print(f"  {prompt_type}: {count}/{total_for_type} ({count/total_for_type*100:.1f}%)")
print()

# Check correctness using extracted answer
def is_correct(row):
    answer = row['extracted_answer']
    gt = row['ground_truth']
    
    if answer == 'INVALID':
        return False
    
    correct_mapping = {
        'Left': 'A',
        'Right': 'B',
        'On top': 'C',
        'Under': 'D'
    }
    
    return answer == correct_mapping.get(gt, None)

df['correct'] = df.apply(is_correct, axis=1)

# Overall accuracy
accuracy = df['correct'].mean()
print(f"Overall Multiple Choice Accuracy: {accuracy * 100:.2f}%")

# Accuracy on VALID answers only (excluding invalid)
valid_df = df[df['extracted_answer'] != 'INVALID']
if len(valid_df) > 0:
    valid_accuracy = valid_df['correct'].mean()
    print(f"Accuracy on valid answers only: {valid_accuracy * 100:.2f}%")
    wandb.log({"valid_only_accuracy": valid_accuracy})

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
wandb.log({"invalid_answer_rate": invalid_count / total_count})

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