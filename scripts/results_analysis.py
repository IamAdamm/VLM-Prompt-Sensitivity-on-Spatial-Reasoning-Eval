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
    Extract a normalized answer from various response formats.
    Returns one of: 'Left', 'Right', 'On top', 'Under', or 'INVALID'
    """
    if not answer or not isinstance(answer, str):
        return 'INVALID'
    
    answer = answer.strip()
    answer_lower = answer.lower()
    
    # Case 1: Direct word answers (what we want from Chain-of-Thought style prompts)
    # Check if answer starts with or equals the target words
    if answer_lower in ['left', 'right', 'under'] or answer_lower.startswith('left') or answer == 'Left':
        return 'Left' if 'left' in answer_lower else answer.capitalize()
    if answer_lower in ['right'] or answer_lower.startswith('right') or answer == 'Right':
        return 'Right'
    if answer_lower in ['on top', 'on top.', 'ontop'] or answer_lower.startswith('on top'):
        return 'On top'
    if answer_lower in ['under', 'under.'] or answer_lower.startswith('under'):
        return 'Under'
    
    # Case 2: Single letter A/B/C/D (with optional period) - Multiple Choice format
    if re.match(r'^[A-Da-d]\.?$', answer):
        letter = answer[0].upper()
        return {'A': 'Left', 'B': 'Right', 'C': 'On top', 'D': 'Under'}.get(letter, 'INVALID')
    
    # Case 3: Check if the answer ENDS with a target word or letter
    # This handles "...the apple is on the Left" or "...A."
    end_match = re.search(r'(Left|Right|On top|Under|[A-D])\.?\s*$', answer, re.IGNORECASE)
    if end_match:
        found = end_match.group(1)
        if found.upper() in 'ABCD':
            return {'A': 'Left', 'B': 'Right', 'C': 'On top', 'D': 'Under'}.get(found.upper(), 'INVALID')
        return found.capitalize() if found.lower() != 'on top' else 'On top'
    
    # Case 4: Model repeated the mapping without answering - INVALID
    if re.match(r'^A\s*=\s*Left', answer) or answer == "A, B, C, D":
        return 'INVALID'
    if re.match(r'^A\s*=\s*\w+,\s*B\s*=', answer):
        return 'INVALID'
    
    # Case 5: Just listed objects without answering - INVALID  
    if re.match(r'^[A-Za-z]+,\s*[A-Za-z]+,\s*[A-Za-z]+', answer) and len(answer) < 100:
        return 'INVALID'
    
    # Case 6: Long descriptive answer - try to extract spatial relationship from text
    # Look for explicit statements about position
    left_patterns = [
        r'\bto the left\b', r'\bon the left\b', r'\bleft side\b', r'\bleft of the\b',
        r'\blocated.*to the left\b', r'\bpositioned.*left\b'
    ]
    right_patterns = [
        r'\bto the right\b', r'\bon the right\b', r'\bright side\b', r'\bright of the\b',
        r'\blocated.*to the right\b', r'\bpositioned.*right\b'
    ]
    top_patterns = [
        r'\bon top of\b', r'\bon the (chair|table|armchair)\b', r'\bresting on\b',
        r'\bsitting on\b', r'\bplaced on the\b', r'\bon the seat\b'
    ]
    under_patterns = [
        r'\bunder the\b', r'\bunderneath\b', r'\bbeneath\b', r'\bbelow the\b'
    ]
    
    # Count matches for each direction
    left_count = sum(1 for p in left_patterns if re.search(p, answer_lower))
    right_count = sum(1 for p in right_patterns if re.search(p, answer_lower))
    top_count = sum(1 for p in top_patterns if re.search(p, answer_lower))
    under_count = sum(1 for p in under_patterns if re.search(p, answer_lower))
    
    # If one direction clearly dominates, use it
    counts = {'Left': left_count, 'Right': right_count, 'On top': top_count, 'Under': under_count}
    max_count = max(counts.values())
    
    if max_count > 0:
        winners = [k for k, v in counts.items() if v == max_count]
        if len(winners) == 1:
            return winners[0]
    
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

# Show Scene Graph CoT specific info if present
if 'scene_graph' in df.columns:
    scene_graph_results = df[df['prompt_label'] == 'Scene Graph CoT']
    if len(scene_graph_results) > 0:
        print("Scene Graph CoT Pipeline Info:")
        print(f"  Total Scene Graph CoT runs: {len(scene_graph_results)}")
        # Check if scene graphs were generated (non-empty)
        valid_scene_graphs = scene_graph_results['scene_graph'].apply(lambda x: x is not None and len(str(x)) > 10 if pd.notna(x) else False).sum()
        print(f"  Valid scene graphs generated: {valid_scene_graphs}/{len(scene_graph_results)}")
        if 'step1_confidence' in df.columns:
            avg_step1_conf = scene_graph_results['step1_confidence'].mean()
            print(f"  Average Step 1 (Scene Graph) confidence: {avg_step1_conf*100:.2f}%")
        print()

# Check correctness using extracted answer
def is_correct(row):
    answer = row['extracted_answer']
    gt = row['ground_truth']
    
    if answer == 'INVALID':
        return False
    
    # Now answers are normalized to 'Left', 'Right', 'On top', 'Under'
    # Ground truth is also in this format
    return answer == gt

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