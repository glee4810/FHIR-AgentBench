import os
import sys
import ast
import json
import re
import multiprocessing as mp
from functools import partial
import yaml
import argparse
import datetime
import csv
from utils import is_reasoning_llm

import pandas as pd
from tqdm import tqdm
import numpy as np
import litellm


from utils.metrics_utils import (
    retrieval_recall,
    retrieval_precision
)
from config import (
    QUESTION_SQL_FHIR_OUTPUT_CSV
)

# Argparser
parser = argparse.ArgumentParser(description="Evaluate FHIR agent answers.")
parser.add_argument("--input",
                    type=str,
                    required=True,
                    help="Path to the input CSV or JSON file with agent results.")
parser.add_argument("--output_dir",
                    type=str,
                    default='output_eval',
                    required=False,
                    help="Path to the output directory.")
parser.add_argument("--model",
                    type=str,
                    default="o4-mini",
                    required=False,
                    help="Model to use for evaluation.")
parser.add_argument("--num_processes",
                    type=int,
                    default=-1,
                    help="Number of processes to use for parallel processing.")
parser.add_argument("--force",
                    action="store_true",
                    help="Force re-evaluation even if results already exist.")
args = parser.parse_args()


# Setup LiteLLM
from utils.core_utils import setup_api_keys

def setup_environment():
    """Setup environment and configuration"""
    if not setup_api_keys():
        raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable or add it to config.yml")
    
    # Configure LiteLLM
    litellm.suppress_debug_info = True
    litellm.set_verbose = False
    
    # Set number of processes
    if args.num_processes == -1:
        args.num_processes = os.cpu_count()

# ============================================================================
# LLM and Processing Functions
# ============================================================================

def call_model(model, prompt):
    """Call LLM with appropriate settings"""
    try:
        model_mapping = {
            "o4-mini": "o4-mini"
        }
        
        litellm_model = model_mapping.get(model, model)
        messages = [{"role": "user", "content": prompt}]
        
        if is_reasoning_llm(model):
            response = litellm.completion(model=litellm_model, messages=messages)
        else:
            response = litellm.completion(model=litellm_model, messages=messages, temperature=0.0)
            
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise ValueError(f"Failed to call model {model}: {str(e)}")


def process_in_parallel(data_tuples, worker_func, desc="Processing", num_processes=1):
    """Process data in parallel or sequentially"""
    if num_processes == 1:
        results = []
        for data_tuple in tqdm(data_tuples, desc=f"{desc} (single-core)"):
            results.append(worker_func(data_tuple))
        return results
    else:
        with mp.Pool(processes=num_processes) as pool:
            results = list(tqdm(
                pool.imap(worker_func, data_tuples),
                total=len(data_tuples),
                desc=f"{desc} (multi-core)"
            ))
        return results


# Worker functions for multiprocessing
def worker_normalize_answer(data_tuple):
    idx, agent_answer, question, model = data_tuple
    try:
        result = normalize_null_answers_with_llm(agent_answer, question, model)
        return idx, result
    except Exception as e:
        print(f"Error processing during normalize_answer: row {idx}: {e}")
        return idx, "question answered"


def worker_check_correctness(data_tuple):
    idx, agent_answer, true_answer, question, model, agent_answer_normalized, error = data_tuple
    null_answer_check = re.sub(r'[^\w\s]', '', agent_answer_normalized).strip().lower()
    try:
        error_mask = (
            pd.isnull(error) or
            (error == "") or
            (error == "nan") or
            (error.strip() == "")
        )
        # Error cases - wrong
        if not error_mask:
            return idx, 0
        # Token exceeded - wrong
        if pd.notnull(agent_answer) and 'Input tokens exceeded' in agent_answer:
            return idx, 0
        # If no answer outputted by both true answer and null answer check, correct
        if null_answer_check == true_answer == "no answer":
            return idx, 1
        # Otherwise if agent answer was none or null, wrong
        if pd.isnull(agent_answer):
            return idx, 0
        # Final check for all other answer types
        result = check_answer_correctness_with_llm(agent_answer, true_answer, question, model)
        return idx, result
    except Exception as e:
        print(f"Error processing during check_correctness: row {idx}: {e}")
        return idx, 0

# ============================================================================
# LLM Evaluation Functions
# ============================================================================

def check_answer_correctness_with_llm(answer: str, ref_answer: str, question: str, model: str = None) -> int:
    prompt = f"""You are a helpful assistant that evaluates whether a model answer to a question is correct, by comparing it to the true answer.

Your task:
- Return 1 if the model answer is correct.
- Return 0 if the model answer is incorrect.
- Never return anything other than 0 or 1.

The model answer may be more verbose or formatted differently from the true answer. Focus on correctness of content, not exact formatting.

---

### Core Rules:

1. Null or no-answer cases  
   - If the true answer is `[]` or `'null'` or 'no answer', this means there is no answer or the answer is effectively 0.  
   - If the model gives any conflicting or non-empty answer, return 0.  
   - If the model also implies no answer, return 1.
   - If the model also implies no answer, or returns 0 for numerical questions, return 1.

2. Yes/No answers 
   - True answers may appear as `[[0]]` (No) or `[[1]]` (Yes), with flexible formatting (e.g., `[0]`, `'[[0]]'` are equivalent).
   - Evaluate based on meaning, not syntax.
   - Ignore differences in variable names or context if the Yes/No meaning aligns.

3. Numerical answers
   - Match on value, rounding both sides to the nearest integer, except for days (which should be rounded to the nearest whole day).
   - Be lenient if the model answer has the true answer in the breakdown but returns a different total aggregated value.
   - Ignore decimal formatting (`1.0` = `1.` = `1`).
   - Ignore units (`1850` = `1850 mL`).

4. Date answers
    - Match on dates, you can ignore time and timezone differences unless specifically stated in the question.
    - A time difference of up to a minute is acceptable, as the values may be rounded.

5. List answers
   - If the true answer is a list (e.g., `['or ebl', 'or urine']`), the model must include all listed values and no extra medical values.  
   - Ignore harmless extra context (e.g., time references, phrasing).

6. Detail and verbosity
   - Extra correct details in the model answer are fine if they align with the true answer.

7. Formatting leniency
   - Be lenient with brackets, quotes, spacing, and style.
   - As long as the model’s answer semantically matches the true answer, return 1.

---

### Examples:

EXAMPLE 1:
- question : Please tell me the sex of patient C  
- true answer : [['F']]  
- model answer : The patient’s sex is female.  
- reasoning : The true answer indicates "female" (F). The model states the patient’s sex is female, which directly matches.  
You will return : 1

---

EXAMPLE 2:  
- question : Compared to last measured on the first icu visit, is the glucose measurement value of patient B less than second to last measured on the first icu visit?  
- true answer : '
- model answer : The patient’s last glucose measurement on the current hospital visit (87 mg/dL) was greater than the first measurement (86 mg/dL).  
- reasoning : The true answer is No (`[[0]]`). The model states the value is greater, which also implies it is not less. This aligns with the true answer meaning.  
You will return : 1

---

EXAMPLE 3:
- question : Are patient A's bicarbonate levels second measured on the first hospital visit less than the levels first measured on the first hospital visit?  
- true answer : [[0]]
- model answer: The second bicarbonate measurement (26 mEq/L at 2161-04-28 02:15) is not less than the first measurement (23 mEq/L at 2161-04-27 12:09).  
- reasoning : The true answer is No (`[[0]]`). The model explicitly says the second value is not less than the first, which matches the true answer.  
You will return : 1

---

EXAMPLE 4:
- question : "Hey, what was patient 10019172's minimum ['fibrinogen, functional'] value in 11/this year?"  
- true answer : [[201.]]  
- model answer: The minimum fibrinogen functional value for patient 10019172 in November of this year is 201.  
- reasoning : The true answer is 201. The model provides the same value (ignoring decimals). They match.  
You will return : 1

---

EXAMPLE 5:  
- question : Show me the total amount of patient's output on this month/02.  
- true answer : [[1850]] 
- model answer: The total patient output on 2110-12-02 was 1850 mL.  
- reasoning : The true answer is 1850. The model answer gives the same value and adds "mL", which is acceptable because units are disregarded.  
You will return : 1

---

EXAMPLE 6:
- question : What was the change in ['platelet count'] in patient 10029484 second measured on the first hospital visit compared to the first value measured on the first hospital visit?
- true answer : [[-38]]
- model answer : The platelet count on the first hospital visit went from 261 on the first measurement to 223 on the second—a drop of 38.
- reasoning : The true answer is -38, indicating a decrease. The model states the count dropped by 38, which matches the true answer meaning.
You will return : 1

---

EXAMPLE 7:
- question : When was patient 10004422's last maximum ['cholesterol, total'] value on their last hospital encounter?
- true answer : [['2111-01-16 07:02:00']]
- model answer : The highest total cholesterol value on the patient’s last hospital encounter was 121 mg/dL, recorded on 2111-01-16.
- reasoning : The true answer is a timestamp. The model provides the same timestamp and mentions the maximum cholesterol value, which aligns with the true answer context. Here there is additional information (the max value) but the answer is still correct. It is okay that the answer does not contain the exact time. 
You will return : 1

--

EXAMPLE 8:
- question : Tell me the total ['acetaminophen'] dose that patient 10037975 has been prescribed?
- true answer : [[1300]]
- model answer : "Total acetaminophen prescribed: 3,300 mg (3.3 g).

Breakdown:
- Acetaminophen IV: 2 prescriptions × 1,000 mg = 2,000 mg  
- Acetaminophen: 2 prescriptions × 650 mg = 1,300 mg"
- reasoning : The true answer only shows the total dose for the basic variation of Acetaminophen, whereas the model provides the total dose for all variations, breaking down the count for the basic one.
You will return : 1

--

### Final Input:
- Question: {question}
- True answer: {ref_answer}
- Model answer: {answer}

### Output:
Return only 0 or 1. Do not explain your reasoning.

"""
    eval_model = model or args.model
    while True:
        response = call_model(eval_model, prompt).strip("'")
        if response in ['0', '1']:
            return int(response)


def normalize_null_answers_with_llm(answer: str, question: str, model: str = None) -> str:
    """Normalize generated answer to reference answer"""
    prompt = f"""
You are a helpful assistant that converts free-text answers into a standardized format.

Your task: Determine if the model's answer successfully answers the question or if it indicates insufficient information.

### Instructions:
1. If the answer shows that the question CANNOT be answered (e.g., not enough data, unknown, missing information), return exactly:
    no answer

2. If the answer DOES answer the question, return exactly:
    question answered

Important:
- Return ONLY one of these two values: 'no answer' or 'question answered'.
- Do NOT include any additional text, punctuation, or explanation.

---

### Input:

Question:
{question}

Answer:
{answer}

---

### Examples:

EXAMPLE 1:
Question: Please tell me the sex of patient C  
Answer: The patient's sex is female.  
Return: question answered

---

EXAMPLE 2:
Question: Compared to last measured on the first ICU visit, is the glucose measurement value of patient B less than second to last measured on the first ICU visit?  
Answer: The patient does not have glucose measurements.
Return: no answer

"""
    eval_model = model or args.model
    return call_model(eval_model, prompt)


# ============================================================================
# Data Processing Functions
# ============================================================================

def extract_relevant_resources_only(row):
    """Extract only resources that match the true FHIR resource types"""
    relevant_resources = []
    if not isinstance(row["true_fhir_ids"], dict):
        return relevant_resources
        
    for resource_type in row["true_fhir_ids"]:
        try:
            agent_resources = row["agent_fhir_resources"]
            if pd.isna(agent_resources) or agent_resources == "" or agent_resources == "nan":
                continue
                
            # Parse agent resources
            if isinstance(agent_resources, str):
                try:
                    rsc_obj = json.loads(agent_resources)
                except json.JSONDecodeError:
                    rsc_obj = ast.literal_eval(agent_resources)
            else:
                rsc_obj = agent_resources
                
            if isinstance(rsc_obj, dict) and resource_type in rsc_obj:
                relevant_resources += rsc_obj[resource_type]
                
        except Exception as e:
            print(f"Error processing resources for {resource_type}: {e}")
            
    return relevant_resources


# ============================================================================
# Main Pipeline Functions
# ============================================================================

def load_and_merge_data():
    """Load input data and merge with ground truth"""
    print("=" * 60)
    print("READING DATA...")
    print("=" * 60)
    
    # Load ground truth
    ground_truth_df = pd.read_csv(QUESTION_SQL_FHIR_OUTPUT_CSV)
    ground_truth_df["true_fhir_ids"] = ground_truth_df["true_fhir_ids"].apply(
        lambda x: ast.literal_eval(x) if pd.notnull(x) else dict()
    )
    
    # Load input data
    if args.input.endswith('.json'):
        with open(args.input, 'r', encoding='utf-8') as f:
            input_df = pd.DataFrame(json.load(f))
        print(f"Read JSON file with {len(input_df)} records")
    else:
        input_df = pd.read_csv(args.input)
        print(f"Read CSV file with {len(input_df)} records")
    
    # Merge data
    cols = [col for col in ground_truth_df.columns 
            if (col not in input_df.columns) or (col == "question_id")]
    eval_df = input_df.merge(ground_truth_df[cols], on="question_id", how="inner")
    
    print(f"Unique questions: {input_df['question_id'].nunique()}")
    print(f"After merge: {eval_df.shape}")
    return eval_df


def filter_evaluation_data(eval_df):
    """Filter evaluation data to remove invalid entries"""
    print("=" * 60)
    print("FILTERING DATA...")
    print("=" * 60)
    
    original_shape = eval_df.shape
    print(f"Original shape: {original_shape}")
    
    # Remove rows without answers
    # eval_df = eval_df[eval_df['agent_answer'].notnull()]
    print(f"# rows without answers: {eval_df['agent_answer'].isnull().sum()}")
    
    # Remove rows with errors (keep NaN and empty strings)
    if 'error' in eval_df.columns:
        error_mask = (eval_df["error"].isnull() | 
                     (eval_df["error"] == "") | 
                     (eval_df["error"] == "nan") | 
                     (eval_df["error"].astype(str).str.strip() == ""))
        
        print(f"# rows with errors: {len(eval_df.loc[~error_mask])}")
    
    if len(eval_df) == 0:
        print("ERROR: No data left after filtering!")
        sys.exit(1)
        
    return eval_df


def calculate_retrieval_metrics(eval_df):
    """Calculate retrieval precision and recall metrics"""
    print("=" * 60)
    print("CALCULATING RETRIEVAL METRICS...")
    print("=" * 60)
    
    # Extract relevant resources and calculate metrics
    eval_df["metric_relevant_resources"] = eval_df.apply(extract_relevant_resources_only, axis=1)
    eval_df["true_fhir_ids_list"] = eval_df["true_fhir_ids"].apply(
        lambda d: sum(d.values(), []) if isinstance(d, dict) else []
    )
    
    # Add resource types column
    eval_df["true_fhir_resource_types"] = eval_df["true_fhir_ids"].apply(
        lambda d: "{" + ", ".join(sorted(d.keys())) + "}" if isinstance(d, dict) else "{}"
    )
    
    # Calculate metrics
    eval_df["recall"] = eval_df.apply(
        lambda row: retrieval_recall(row["metric_relevant_resources"], row["true_fhir_ids_list"]), axis=1)
    eval_df["precision"] = eval_df.apply(
        lambda row: retrieval_precision(row["metric_relevant_resources"], row["true_fhir_ids_list"], method="continuous"), axis=1)
    
    print(f"Retrieval Precision: {eval_df['precision'].mean():.4f}")
    print(f"Retrieval Recall:    {eval_df['recall'].mean():.4f}")
    return eval_df


def calculate_answer_metrics(eval_df, num_processes=1):
    """Calculate answer correctness metrics using LLM evaluation"""
    print("=" * 60)
    print("CALCULATING ANSWER METRICS...")
    print("=" * 60)
    
    # Set up true answers
    eval_df['true_answer_extracted'] = eval_df["true_answer"]
    eval_df.loc[eval_df["true_fhir_resource_types"] == "{}", "true_answer_extracted"] = "no answer"

    # Normalize answers
    print("Normalizing answers...")
    normalize_data = [(idx, row["agent_answer"], row["question"], args.model) 
                     for idx, row in eval_df.iterrows()]
    normalize_results = process_in_parallel(normalize_data, worker_normalize_answer, 
                                          "Normalizing answers", num_processes)
    normalize_results.sort(key=lambda x: x[0])
    eval_df["agent_answer_normalized"] = [result[1] for result in normalize_results]

    # Check correctness
    print("Checking correctness...")
    correctness_data = [(idx, row["agent_answer"], row["true_answer_extracted"], row["question"], args.model, row["agent_answer_normalized"], row["error"]) 
                       for idx, row in eval_df.iterrows()]
    correctness_results = process_in_parallel(correctness_data, worker_check_correctness, 
                                            "Checking correctness", num_processes)
    correctness_results.sort(key=lambda x: x[0])
    eval_df["answer_correctness"] = [result[1] for result in correctness_results]
    
    return eval_df


def save_results(eval_df):
    """Save evaluation results to JSON file"""
    print("=" * 60)
    print("SAVING RESULTS...")
    print("=" * 60)
    
    save_cols = [
        "question_id", "question", "assumption", "question_with_context", 
        "true_answer", "agent_answer", "answer_correctness",
        "trace", "true_answer_extracted", "agent_answer_normalized", "usage",
        "precision", "recall", "true_fhir_resource_types", "template"
    ]

    os.makedirs(args.output_dir, exist_ok=True)
    
    input_base = os.path.splitext(os.path.basename(args.input))[0]
    output_path = os.path.join(args.output_dir, f'{input_base}_eval.json')
    
    eval_df[save_cols].to_json(output_path, orient='records', indent=2)
    print(f"Results saved to: {output_path}")
    return output_path




def save_performance_summary_to_csv(eval_df, input_file, output_eval_file, output_dir="output_eval"):
    """Save performance summary to a CSV file, one row per resource type (including Overall)"""
    summary_csv = os.path.join(output_dir, "performance_summary.csv")
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rows = []

    # Overall metrics
    total_questions = len(eval_df)
    correct_answers = eval_df['answer_correctness'].sum()
    accuracy = correct_answers / total_questions if total_questions > 0 else 0
    avg_precision = eval_df['precision'].mean()
    avg_recall = eval_df['recall'].mean()
    f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0

    rows.append({
        "date": now,
        "input_file": input_file,
        "output_eval_file": output_eval_file,
        "fhir_resource_type": "Overall",
        "questions": total_questions,
        "precision": avg_precision,
        "recall": avg_recall,
        "f1_score": f1_score,
        "accuracy": accuracy
    })

    # Resource type breakdown
    if 'true_fhir_resource_types' in eval_df.columns:
        resource_performance = eval_df.groupby('true_fhir_resource_types').agg({
            'answer_correctness': ['count', 'sum', 'mean'],
            'precision': 'mean',
            'recall': 'mean'
        }).round(4)
        for resource_type in resource_performance.index:
            count = resource_performance.loc[resource_type, ('answer_correctness', 'count')]
            correct = resource_performance.loc[resource_type, ('answer_correctness', 'sum')]
            acc = resource_performance.loc[resource_type, ('answer_correctness', 'mean')]
            prec = resource_performance.loc[resource_type, ('precision', 'mean')]
            rec = resource_performance.loc[resource_type, ('recall', 'mean')]
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
            rows.append({
                "date": now,
                "input_file": input_file,
                "output_eval_file": output_eval_file,
                "fhir_resource_type": resource_type,
                "questions": count,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "accuracy": acc
            })

    # Write or append to CSV
    os.makedirs(output_dir, exist_ok=True)
    write_header = not os.path.exists(summary_csv)
    fieldnames = [
        "date", "input_file", "output_eval_file", "fhir_resource_type",
        "questions", "precision", "recall", "f1_score", "accuracy"
    ]
    with open(summary_csv, "a", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)

def print_performance_summary(eval_df):
    """Print comprehensive performance summary and save to CSV"""
    print("=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    
    total_questions = len(eval_df)
    
    # Answer correctness metrics
    correct_answers = eval_df['answer_correctness'].sum()
    accuracy = correct_answers / total_questions if total_questions > 0 else 0
    
    # Retrieval metrics
    avg_precision = eval_df['precision'].mean()
    avg_recall = eval_df['recall'].mean()
    f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
    
    print(f"Total Questions Evaluated: {total_questions}")
    print()
    print("ANSWER CORRECTNESS:")
    print(f"  Correct Answers: {correct_answers}/{total_questions}")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print()
    print("RETRIEVAL PERFORMANCE:")
    print(f"  Average Precision: {avg_precision:.4f}")
    print(f"  Average Recall: {avg_recall:.4f}")
    print(f"  F1 Score: {f1_score:.4f}")
    print()
    
    # Performance by resource type
    if 'true_fhir_resource_types' in eval_df.columns:
        print("PERFORMANCE BY RESOURCE TYPE:")
        resource_performance = eval_df.groupby('true_fhir_resource_types').agg({
            'answer_correctness': ['count', 'sum', 'mean'],
            'precision': 'mean',
            'recall': 'mean'
        }).round(4)
        
        for resource_type in resource_performance.index:
            count = resource_performance.loc[resource_type, ('answer_correctness', 'count')]
            correct = resource_performance.loc[resource_type, ('answer_correctness', 'sum')]
            acc = resource_performance.loc[resource_type, ('answer_correctness', 'mean')]
            prec = resource_performance.loc[resource_type, ('precision', 'mean')]
            rec = resource_performance.loc[resource_type, ('recall', 'mean')]
            
            print(f"  {resource_type}:")
            print(f"    Questions: {count}, Correct: {correct}, Accuracy: {acc:.4f}")
            print(f"    Precision: {prec:.4f}, Recall: {rec:.4f}")
        print()
    
    print("=" * 60)


def check_existing_results():
    """Check if evaluation results already exist"""
    input_base = os.path.splitext(os.path.basename(args.input))[0]
    output_path = os.path.join(args.output_dir, f'{input_base}_eval.json')
    
    if os.path.exists(output_path) and not args.force:
        print("=" * 60)
        print("EXISTING RESULTS FOUND!")
        print("=" * 60)
        print(f"Results file already exists: {output_path}")
        print("Use --force to re-evaluate or press Enter to view existing results.")
        
        response = input("Do you want to re-evaluate? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("Loading existing results...")
            existing_df = pd.read_json(output_path)
            print_performance_summary(existing_df)
            return True, existing_df
    elif os.path.exists(output_path) and args.force:
        print(f"Force mode: Overwriting existing results at {output_path}")
    
    return False, None


def main():
    """Main evaluation pipeline"""
    setup_environment()
    
    # Check if results already exist
    exists, existing_df = check_existing_results()
    if exists:
        # If loaded from existing, we don't know the output file path, so skip CSV save
        print_performance_summary(existing_df)
        return
    
    # Run full evaluation pipeline
    eval_df = load_and_merge_data()
    eval_df = filter_evaluation_data(eval_df)
    eval_df = calculate_retrieval_metrics(eval_df)
    eval_df = calculate_answer_metrics(eval_df, args.num_processes)
    output_path = save_results(eval_df)
    
    # Print performance summary and save to CSV
    print_performance_summary(eval_df)

    # Save summary to CSV
    save_performance_summary_to_csv(eval_df, args.input, output_path, args.output_dir)    
    print("=" * 60)
    print("EVALUATION COMPLETE!")
    print("=" * 60)
    print()



if __name__ == "__main__":
    main()