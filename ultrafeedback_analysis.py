"""
RLVR Analyzer - UltraFeedback Analysis (2024 Dataset)
Modern AI feedback dataset - more relevant than HH-RLHF!

Run this locally on your machine
"""

from datasets import load_dataset
import pandas as pd
import numpy as np
from scipy import stats
import json

print("=" * 80)
print("UltraFeedback Dataset Analysis (2024)")
print("AI Feedback from GPT-4 | 64K prompts | 256K responses")
print("=" * 80)

print("\n📥 Loading UltraFeedback dataset...")
# Load the binarized version (already has chosen/rejected pairs)
dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized")

print(f"\n✅ Dataset loaded!")
print(f"Train examples: {len(dataset['train_prefs'])}")
print(f"Test examples: {len(dataset['test_prefs'])}")

print("\n" + "=" * 80)
print("STEP 1: Analyzing Chosen vs Rejected Responses")
print("=" * 80)

# Extract chosen and rejected from preference pairs
chosen_responses = []
rejected_responses = []

print("\n⏳ Processing examples...")
for i, example in enumerate(dataset['train_prefs']):
    if i % 5000 == 0:
        print(f"Processed {i}/{len(dataset['train_prefs'])} examples...")
    
    # UltraFeedback format: messages list with 'chosen' and 'rejected'
    chosen_responses.append(example['chosen'][-1]['content'])  # Last message is assistant response
    rejected_responses.append(example['rejected'][-1]['content'])

print(f"\n✅ Extracted {len(chosen_responses)} preference pairs")

print("\n" + "=" * 80)
print("STEP 2: Length Analysis")
print("=" * 80)

chosen_lengths = [len(response) for response in chosen_responses]
rejected_lengths = [len(response) for response in rejected_responses]

print("\n📊 LENGTH STATISTICS:")
print(f"\nChosen responses:")
print(f"  Mean: {np.mean(chosen_lengths):.1f} characters")
print(f"  Median: {np.median(chosen_lengths):.1f}")
print(f"  Std Dev: {np.std(chosen_lengths):.1f}")
print(f"  Min: {np.min(chosen_lengths)}, Max: {np.max(chosen_lengths)}")

print(f"\nRejected responses:")
print(f"  Mean: {np.mean(rejected_lengths):.1f} characters")
print(f"  Median: {np.median(rejected_lengths):.1f}")
print(f"  Std Dev: {np.std(rejected_lengths):.1f}")
print(f"  Min: {np.min(rejected_lengths)}, Max: {np.max(rejected_lengths)}")

ratio = np.mean(chosen_lengths) / np.mean(rejected_lengths)
print(f"\n🎯 KEY FINDING: Chosen responses are {ratio:.2f}x longer")

print("\n" + "=" * 80)
print("STEP 3: Statistical Significance")
print("=" * 80)

t_statistic, p_value = stats.ttest_ind(chosen_lengths, rejected_lengths)
pooled_std = np.sqrt((np.std(chosen_lengths)**2 + np.std(rejected_lengths)**2) / 2)
cohens_d = (np.mean(chosen_lengths) - np.mean(rejected_lengths)) / pooled_std

print(f"\n📈 T-TEST:")
print(f"  t-statistic: {t_statistic:.4f}")
print(f"  p-value: {p_value:.2e}")
print(f"  {'✅ HIGHLY SIGNIFICANT (p < 0.001)' if p_value < 0.001 else '⚠️ Not significant'}")

print(f"\n📏 EFFECT SIZE (Cohen's d): {cohens_d:.3f}")
if cohens_d > 0.8:
    print(f"   ✅ Large effect (> 0.8) - Strong verbosity bias confirmed!")
elif cohens_d > 0.5:
    print(f"   Medium effect")
else:
    print(f"   Small effect")

print("\n" + "=" * 80)
print("STEP 4: Prompt Type Analysis (Bonus!)")
print("=" * 80)

# UltraFeedback has prompts from different sources
# Let's analyze first 100 to categorize
print("\n⏳ Sampling 100 prompts for manual categorization...")
prompt_types = {
    'code': 0,
    'math': 0, 
    'factual_qa': 0,
    'creative': 0,
    'advice': 0,
    'other': 0
}

# Simple keyword-based categorization (you can improve this)
for i, example in enumerate(dataset['train_prefs'][:100]):
    try:
        # Get the prompt - it's the first message in the chosen conversation
        if isinstance(example['chosen'], list) and len(example['chosen']) > 0:
            prompt = example['chosen'][0]['content'].lower()  # First message is the prompt
        elif isinstance(example.get('prompt'), str):
            # Some examples might have prompt directly
            prompt = example['prompt'].lower()
        else:
            # Skip if format is unexpected
            prompt_types['other'] += 1
            continue
    except (KeyError, TypeError, IndexError) as e:
        # Skip examples with unexpected structure
        prompt_types['other'] += 1
        continue
    
    if any(word in prompt for word in ['code', 'python', 'javascript', 'function', 'program']):
        prompt_types['code'] += 1
    elif any(word in prompt for word in ['calculate', 'math', 'solve', 'equation', 'number']):
        prompt_types['math'] += 1
    elif any(word in prompt for word in ['what is', 'who is', 'when did', 'where is', 'define']):
        prompt_types['factual_qa'] += 1
    elif any(word in prompt for word in ['write a story', 'poem', 'create', 'imagine']):
        prompt_types['creative'] += 1
    elif any(word in prompt for word in ['should i', 'advice', 'recommend', 'suggest', 'help me']):
        prompt_types['advice'] += 1
    else:
        prompt_types['other'] += 1

print("\n📋 PROMPT TYPE DISTRIBUTION (sample of 100):")
for ptype, count in prompt_types.items():
    print(f"  {ptype.capitalize()}: {count}%")

print("\n" + "=" * 80)
print("STEP 5: Save Results")
print("=" * 80)

results = {
    "dataset_name": "UltraFeedback (GPT-4 AI Feedback)",
    "total_examples": len(chosen_responses),
    "data_source": "64K prompts from UltraChat, ShareGPT, Evol-Instruct, TruthfulQA, FalseQA, FLAN",
    "annotation_method": "GPT-4 (AI Feedback on 4 aspects: instruction-following, truthfulness, honesty, helpfulness)",
    "year": "2024",
    
    "chosen_stats": {
        "mean": float(np.mean(chosen_lengths)),
        "median": float(np.median(chosen_lengths)),
        "std": float(np.std(chosen_lengths)),
        "min": int(np.min(chosen_lengths)),
        "max": int(np.max(chosen_lengths))
    },
    "rejected_stats": {
        "mean": float(np.mean(rejected_lengths)),
        "median": float(np.median(rejected_lengths)),
        "std": float(np.std(rejected_lengths)),
        "min": int(np.min(rejected_lengths)),
        "max": int(np.max(rejected_lengths))
    },
    "analysis": {
        "length_ratio": float(ratio),
        "t_statistic": float(t_statistic),
        "p_value": float(p_value),
        "cohens_d": float(cohens_d),
        "is_significant": bool(p_value < 0.001),
        "interpretation": "Large effect" if cohens_d > 0.8 else "Medium effect" if cohens_d > 0.5 else "Small effect"
    },
    "prompt_type_distribution": prompt_types,
    
    # Sample data for visualization
    "chosen_sample": chosen_lengths[:200],
    "rejected_sample": rejected_lengths[:200]
}

output_file = "ultrafeedback_analysis_results.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Results saved to: {output_file}")

print("\n" + "=" * 80)
print("KEY INSIGHTS FOR YOUR ANALYZER:")
print("=" * 80)
print("""
1. UltraFeedback uses GPT-4 for annotation (AI feedback, not human)
   → Shows you understand modern RLAIF approach!
   
2. Larger & more diverse than HH-RLHF
   → 64K prompts vs 160K pairs, but from 6 different sources
   
3. Has fine-grained scores on 4 dimensions
   → You can analyze which aspect has most bias
   
4. More relevant for 2025 ByteDance interview
   → "I analyzed the UltraFeedback dataset, which uses GPT-4 annotations..."
   → Shows you're current with AI feedback trends
   
5. Can still analyze same issues:
   ✅ Verbosity bias
   ✅ Inconsistency patterns
   ✅ Cost optimization (RLHF vs RLVR)
   ✅ Process improvements
""")

print("\n🎉 ANALYSIS COMPLETE!")
print("=" * 80)
