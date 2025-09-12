import os
import multiprocessing as mp
from tqdm import tqdm
import argparse
import json
import pandas as pd
from utils import (
    validate_config,
    read_input_data, 
    read_intermediate_results, 
    curate_input_dataset,
    run_agent_with_input,
    check_tool_credentials
)

parser = argparse.ArgumentParser(description="Evaluate FHIR agent answers and save results.")
parser.add_argument("--model",
                    type=str,
                    default="o4-mini",
                    help="Model to use (e.g. 'o4-mini', 'gemini/gemini-2.5-flash')"
                    )
parser.add_argument("--base_url",
                    type=str,
                    default=None,
                    help="Base URL for vllm (e.g. http://localhost:8000/v1)"
                    )
parser.add_argument("--input",
                    type=str,
                    help="Input CSV file path",
                    default='final_dataset/questions_answers_sql_fhir.csv'
                    )
parser.add_argument("--output",
                    type=str,
                    default=None,
                    help="Output JSON file path (optional, will be auto-generated if not provided)",
                    )
parser.add_argument("--date",
                    type=str,
                    default=None,
                    help="Date string for output file naming (optional)"
                    )
parser.add_argument("--agent_strategy",
                    required=True,
                    choices=["single_turn_request", "single_turn_resource", "single_turn_code_resource", "multi_turn_resource", "multi_turn_code_resource"],
                    type=str,
                    help="Which agent strategy to use"
                    )
parser.add_argument("--add_patient_fhir_id",
                    type=bool,
                    default=True,
                    help="Add patient FHIR ID to the input data"
                    )
parser.add_argument("--num_processes",
                    type=int,
                    default=-1,
                    help="Number of processes to use"
                    )
parser.add_argument("--save_interval",
                    type=int,
                    default=10,
                    help="Save intermediate results every X rows"
                    )
parser.add_argument("--verbose",
                    action="store_true",
                    default=True,
                    help="Enable verbose output for debugging"
                    )
parser.add_argument("--enable_cache",
                    type=bool,
                    default=True,
                    help="Enable tool caching to speed up repeated FHIR requests"
                    )

args = parser.parse_args()

if args.num_processes == -1:
    args.num_processes = os.cpu_count()

# Validate config
validate_config(args.model, args.base_url)

# Prepare input tuples for multiprocessing
def prepare_multiprocessing_inputs(remaining_inputs, agent_strategy, model, verbose, base_url, enable_cache):
    return [(input_data, agent_strategy, model, verbose, base_url, enable_cache) for input_data in remaining_inputs]


def save_results_json(df, output_file):
    """Save DataFrame to JSON with basic error handling."""
    try:        
        df_copy = df.copy()
        records = df_copy.to_dict('records')
        
        temp_file = output_file + '.tmp'
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2, default=str)
        
        try:
            with open(temp_file, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
                print(f"✅ JSON validation passed: {sum([l['agent_answer'] is not None for l in loaded_data])} records saved")
        except Exception as validation_error:
            print(f"❌ JSON validation failed: {validation_error}")
            print("Critical error: Cannot verify saved data integrity")
            if os.path.exists(temp_file):
                os.remove(temp_file)
            exit(1)
            
        os.rename(temp_file, output_file)
        
    except Exception as e:
        print(f"❌ Error saving JSON: {e}")
        temp_file = output_file + '.tmp'
        if os.path.exists(temp_file):
            os.remove(temp_file)
        print("Critical error: Cannot save results safely")
        exit(1)

if __name__ == "__main__":
    # Prepare for multiprocessing
    input_file = args.input
    output_file = args.output
    
    # Set JSON output file path
    if args.output is None:
        base_name = os.path.splitext(output_file)[0]
        output_json_file = f"{base_name}.json"
    else:
        output_json_file = args.output
    if 'gemini/' in args.model: # For Gemini models, remove 'gemini/' from the output file path
        output_json_file = output_json_file.replace('gemini/', '')
    if 'Qwen/' in args.model: # For Qwen models, remove 'Qwen/' from the output file path
        output_json_file = output_json_file.replace('Qwen/', '')
    if 'meta-llama/' in args.model: # For Meta-Llama models, remove 'meta-llama/' from the output file path
        output_json_file = output_json_file.replace('meta-llama/', '')

    # Read input data
    print(
        """
        ************************************************************
        READING DATA...
        ************************************************************
        """
    )
    # Read intermediate data, if it exists
    # if not os.path.exists(os.path.dirname(output_file)):
    #     os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if not os.path.exists(os.path.dirname(output_json_file)):
        os.makedirs(os.path.dirname(output_json_file), exist_ok=True)

    # Try to read results from JSON
    results_df = read_intermediate_results(output_json_file)
    if results_df.empty:
        results_df = read_input_data(input_file)
        for col in ["agent_answer", "agent_fhir_resources", "trace", "error", "usage"]:
            results_df[col] = None
    elif len(results_df) != len(read_input_data(input_file)):
        results_df = read_input_data(input_file).merge(
            results_df[["question_id", "agent_answer", "agent_fhir_resources", "trace", "error", "usage"]], 
            on="question_id", 
            how="left"
        )

    # Add question_with_context for all rows (if not already present)
    all_inputs = curate_input_dataset(results_df, args.add_patient_fhir_id)
    results_df["question_with_context"] = all_inputs

    # Only subset to data we don't have existing results for to run
    results_df = results_df[['question_id', 'question', 'true_answer', 'assumption', 'question_with_context', 'patient_fhir_id', 'agent_answer', 'agent_fhir_resources', 'trace', 'error', 'usage']]
    remaining_data = results_df.loc[results_df["agent_answer"].isnull()].copy()
    processed_count = results_df["agent_answer"].notnull().sum()
    print(
        "Number of rows already processed prior:",
        processed_count
    )
    print(f"Number of rows to process: {len(remaining_data)}")
    remaining_inputs = remaining_data["question_with_context"].tolist()

    print(
        """
        ************************************************************
        RUNNING AGENT...
        ************************************************************
        """
    )

    # Checking tool credentials
    check_tool_credentials()

    num_processes = min(args.num_processes, len(remaining_inputs))
    if num_processes == 1:
        for i, input_data in enumerate(tqdm(remaining_inputs, total=len(remaining_inputs))):
            processed_output = run_agent_with_input((input_data, args.agent_strategy, args.model, args.verbose, args.base_url, args.enable_cache))
            
            # Update the corresponding row in the DataFrame
            for key, value in processed_output.items():
                results_df.at[remaining_data.index[i], key] = value

            # Save intermediate results every X rows
            if i % args.save_interval == 0:
                save_results_json(results_df, output_json_file)
    else:
        # Prepare inputs for multiprocessing
        input_tuples = prepare_multiprocessing_inputs(remaining_inputs, args.agent_strategy, args.model, args.verbose, args.base_url, args.enable_cache)
        
        with mp.Pool(processes=num_processes) as pool:
            for i, processed_output in enumerate(
                tqdm(
                    pool.imap(run_agent_with_input, input_tuples),
                    total=len(remaining_inputs)
                )
            ):
                # Update the corresponding row in the DataFrame
                for key, value in processed_output.items():
                    results_df.at[remaining_data.index[i], key] = value

                # Save intermediate results every X rows
                if i % args.save_interval == 0:
                    save_results_json(results_df, output_json_file)

    # Save final results
    print(f"Saving final results to JSON: {output_json_file}")
    save_results_json(results_df, output_json_file)
    
    print("Results saved successfully in JSON formats!")
