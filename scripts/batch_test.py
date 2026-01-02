import os
import json
import torch
import torch.nn.functional as F
from datetime import datetime
from PIL import Image
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token

# Define paths and device
project_root = '/data/llava/LLaVA'
prompts = os.path.join(project_root, 'prompts/prompts.json')
images = os.path.join(project_root, 'images')
results_file = os.path.join(project_root, 'results/results.json')
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "/data/llava/models/llava-v1.5-7b"

# Load model, tokenizer, & image processor
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name="llava-v1.5-7b",
    device=device
)

# Set model to eval
model.eval()

# Load prompts as read-only
with open(prompts, 'r') as f:
    prompt_data = json.load(f)

# Array to store results
results = []


def run_single_inference(image_tensor, image_sizes, prompt_text, max_tokens=128):
    """Run a single inference and return the answer, confidence, and raw output."""
    # Build conversation
    conv = conv_templates["llava_v1"].copy() 
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + prompt_text)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Tokenize prompt with image token
    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt"
    ).unsqueeze(0).to(device)

    # Generate response with scores for confidence calculation
    with torch.inference_mode():
        outputs = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            max_new_tokens=max_tokens,
            do_sample=False,
            use_cache=True,
            output_scores=True,
            return_dict_in_generate=True
        )
    
    # Get the generated tokens and scores
    generated_ids = outputs.sequences
    scores = outputs.scores
    
    # Calculate confidence
    confidence = None
    if scores and len(scores) > 0:
        first_token_logits = scores[0][0]
        probs = F.softmax(first_token_logits, dim=-1)
        new_token_start_idx = input_ids.shape[1]
        if generated_ids.shape[1] > new_token_start_idx:
            first_generated_token = generated_ids[0, new_token_start_idx]
            confidence = probs[first_generated_token].item()
        else:
            first_generated_token = torch.argmax(probs).item()
            confidence = probs[first_generated_token].item()
    
    answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
    return answer, confidence


def run_two_step_inference(image_tensor, image_sizes, step1_prompt, step2_prompt):
    """Run a two-step inference pipeline (Scene Graph CoT style)."""
    # Step 1: Generate scene graph
    scene_graph, step1_confidence = run_single_inference(
        image_tensor, image_sizes, step1_prompt, max_tokens=512
    )
    
    # Step 2: Use scene graph context to generate answer
    # Combine the scene graph output with step 2 prompt
    step2_with_context = f"Scene Graph from previous analysis:\n{scene_graph}\n\n{step2_prompt}"
    
    final_answer, step2_confidence = run_single_inference(
        image_tensor, image_sizes, step2_with_context, max_tokens=64
    )
    
    return {
        'scene_graph': scene_graph,
        'final_answer': final_answer,
        'step1_confidence': step1_confidence,
        'step2_confidence': step2_confidence
    }


# Process each prompt entry
for entry in prompt_data:
    # Get the list of images and ground truths for this entry
    image_list = entry.get('image', [])
    ground_truth_map = entry.get('ground_truth', {})
    prompts_list = entry.get('prompts', [])
    
    # Process each image specified in this entry
    for current_image in image_list:
        image_path = os.path.join(images, current_image)
        
        # Load and process image
        image = Image.open(image_path).convert('RGB')
        image_tensor = process_images([image], image_processor, model.config)[0]
        image_tensor = image_tensor.unsqueeze(0).to(device).half()
        image_sizes = [image.size]
        
        # Get ground truth for this specific image
        ground_truth_label = ground_truth_map.get(current_image, None)
        
        # Process each prompt for the current image
        for prompt_entry in prompts_list:
            prompt_label = prompt_entry.get('label', '')
            
            # Check if this is a multi-step prompt (Scene Graph CoT)
            if 'steps' in prompt_entry:
                # Two-step pipeline
                steps = prompt_entry['steps']
                step1_prompt = steps[0]['prompt']
                step2_prompt = steps[1]['prompt']
                
                result = run_two_step_inference(
                    image_tensor, image_sizes, step1_prompt, step2_prompt
                )
                
                # Store result with both steps info
                results.append({
                    'image': current_image,
                    'prompt': f"Step 1: {step1_prompt}\n\nStep 2: {step2_prompt}",
                    'prompt_label': prompt_label,
                    'scene_graph': result['scene_graph'],
                    'answer': result['final_answer'],
                    'confidence': result['step2_confidence'],
                    'step1_confidence': result['step1_confidence'],
                    'ground_truth': ground_truth_label,
                    'timestamp': str(datetime.now())
                })
            else:
                # Single-step prompt (Multiple Choice, Chain-of-Thought, Direct Question)
                prompt_text = prompt_entry.get('prompt', '')
                
                answer, confidence = run_single_inference(
                    image_tensor, image_sizes, prompt_text
                )
                
                # Store result
                results.append({
                    'image': current_image,
                    'prompt': prompt_text,
                    'prompt_label': prompt_label,
                    'answer': answer,
                    'confidence': confidence,
                    'ground_truth': ground_truth_label,
                    'timestamp': str(datetime.now())
                })

# Save results to results.json
with open(results_file, 'w') as f:
    json.dump(results, f, indent=4)

print(f"Finished. Results saved to {results_file}")