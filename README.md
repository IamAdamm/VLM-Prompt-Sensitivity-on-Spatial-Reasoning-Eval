A clean README you can use (tailored to what you actually did)

You can copy this almost 1:1.

⸻

Prompt Sensitivity Analysis in Vision-Language Models

A Case Study Using LLaVA

Overview

This project investigates prompt sensitivity in Vision-Language Models (VLMs), focusing on how wording choices, option ordering, and multiple-choice formatting affect model accuracy.

Using the LLaVA model as a representative VLM, we conduct controlled experiments on spatial reasoning tasks (e.g., left/right, above/below) and analyze how small prompt variations can lead to large performance differences — including systematic failures below random chance.

⸻

Research Questions
	•	How sensitive is a VLM’s accuracy to:
	•	Multiple-choice wording?
	•	Option shuffling?
	•	Semantically similar spatial terms (e.g., above vs under)?
	•	Can prompt variations induce systematic bias rather than random error?
	•	Do certain prompt structures cause the model to rely on linguistic heuristics instead of visual grounding?

⸻

Experimental Setup

Model
	•	LLaVA (pretrained)
	•	Used in inference-only mode
	•	No finetuning or training performed

Access Method
	•	Local inference via:
	•	Gradio web interface (localhost)
	•	LLaVA CLI / evaluation scripts

Task
	•	Image-based spatial reasoning
	•	Multiple-choice questions with controlled prompt variants:
	•	Fixed order
	•	Shuffled order
	•	Lexical variations (“above”, “below”, “under”)

⸻

Prompt Variants Tested

Examples of tested configurations include:
	•	Standard multiple-choice
	•	Multiple-choice with shuffled answer options
	•	Multiple-choice with semantically varied spatial terms
	•	Combined shuffling + wording variation

Each variant was evaluated independently to isolate its effect on accuracy.

⸻

Key Findings (Summary)
	•	Accuracy varies significantly across prompt formulations
	•	Certain combinations (e.g., shuffled options + “above”) led to accuracy far below random chance
	•	This suggests systematic heuristic bias, not simple confusion
	•	Models may prioritize textual patterns and option position over visual grounding

⸻

Interpretation

Results indicate that:
	•	Vision-language models can apply consistent but incorrect linguistic heuristics
	•	Small prompt changes can dramatically alter behavior
	•	High-level performance metrics may hide fragile reasoning mechanisms

⸻

Limitations & Future Work
	•	Only multiple-choice formats tested so far
	•	Chain-of-Thought and free-form reasoning not yet evaluated
	•	Future experiments will:
	•	Compare CoT vs non-CoT prompting
	•	Analyze attention to image tokens
	•	Extend evaluation to additional VLMs

⸻

Model Attribution

This project uses LLaVA, developed by Liu et al.
	•	Official repository: https://github.com/haotian-liu/LLaVA
	•	Papers:
	•	Visual Instruction Tuning (NeurIPS 2023)
	•	Improved Baselines with Visual Instruction Tuning (2023)

No modifications were made to the LLaVA model or training pipeline.

⸻

License

This project follows the license terms of the original LLaVA model and datasets.
See the official LLaVA repository for full license details.