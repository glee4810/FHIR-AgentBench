import pandas as pd
import json
import os
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import custom modules
from config import (
    DATA_PATHS, UNPROCESSED_DB_PATH, PROCESSED_DB_PATH, 
    QUESTION_ANSWER_OUTPUT_CSV, VALUE_MAPPING_FILE, REMOVE_TEMPLATES
)
from utils.fhir_utils import (
    DatabaseManager, ValueMappingProcessor, QuestionProcessor,
    AssumptionGenerator, load_data_efficiently, apply_anchor_replacements,
    get_db_connection
)
from utils.sql_to_fhir_utils import create_unique_table_id, convert_table_id_to_fhir_id
from utils import extract_patient_id_from_fhir_json
from create_db import unprocessed_db


def setup_databases():
    """Set up database connections and verify integrity."""
    db = unprocessed_db(out_dir="data", db_name="mimic_iv_unprocessed", generate=True, lower=True)
    raw_db = unprocessed_db(out_dir="data", db_name="mimic_iv_unprocessed_raw", generate=True, lower=False)
    db.quick_check()
    raw_db.quick_check()
    return db, raw_db


def load_and_prepare_questions():
    """Load and prepare questions data efficiently."""
    full_df = load_data_efficiently(DATA_PATHS)
    print(f"Loaded {full_df.shape[0]} questions")
    
    answerable_df = full_df.copy()
    answerable_df = answerable_df.loc[~answerable_df["query"].str.contains("cost"), :].copy()
    print(f"Shape after removing 'cost' related queries: {answerable_df.shape}")
    answerable_df = answerable_df.loc[~answerable_df["question"].str.lower().str.contains("insurance"), :].copy()
    print(f"Shape after removing 'insurance' related questions: {answerable_df.shape}")
    answerable_df = answerable_df.loc[~answerable_df["query"].str.contains("diagnoses_icd.charttime"), :].copy()
    print(f"Shape after removing questions using 'diagnoses_icd.charttime' in the query: {answerable_df.shape}")
    
    answerable_df["main_table_name"] = answerable_df["query"].apply(lambda x: x.split("FROM ")[1].split(" ")[0])
    answerable_df = (answerable_df.loc[~answerable_df["main_table_name"].isin(["d_icd_diagnoses", "d_icd_procedures"]), :].drop(["main_table_name"], axis=1).copy())

    print(f"Shape after removing questions with d_icd_diagnoses or d_icd_procedures as the main search table in the query: {answerable_df.shape}")

    answerable_df = answerable_df.loc[~answerable_df["template"].isin(REMOVE_TEMPLATES), :].copy()
    print(f"Shape after removing other faulty question templates: {answerable_df.shape}")

    # Extract subject_id and filter
    answerable_df["subject_id"] = answerable_df["val_dict"].apply(extract_patient_id_from_fhir_json)
    answerable_df = answerable_df.loc[answerable_df["subject_id"].notnull(), :].copy()
    print(f"Shape after removing questions without subject_id: {answerable_df.shape}")
    
    return answerable_df


def build_patient_dataframe():
    """Build and process patient dataframe."""
    with get_db_connection(UNPROCESSED_DB_PATH) as conn:
        patient_df = pd.read_sql_query("SELECT * FROM patients", conn)
    
    # Create FHIR IDs
    patient_df["table_id"] = patient_df.apply(lambda row: create_unique_table_id(row, "patients"), axis=1)
    patient_df["fhir_resource"] = "Patient"
    patient_df["fhir_id"] = patient_df.apply(lambda row: convert_table_id_to_fhir_id("Patient", row["table_id"]), axis=1)
    patient_df.drop_duplicates(subset=["table_id", "fhir_id"], inplace=True)
    
    return patient_df


def create_sample_mapping(data_paths):
    """Create id to sample mapping efficiently."""
    all_samples = []
    for file_path in data_paths:
        with open(f"{file_path}/annotated.json", "r") as f:
            all_samples.extend(json.load(f))
    return {q['id']: q for q in all_samples}


def build_value_mappings():
    """Build value mappings efficiently using batch operations."""
    db_manager = DatabaseManager(PROCESSED_DB_PATH, UNPROCESSED_DB_PATH)
    
    # Basic mappings
    tables_and_columns = [
        ('patients', 'gender'), ('d_icd_procedures', 'long_title'), ('d_icd_diagnoses', 'long_title'),
        ('prescriptions', 'drug'), ('prescriptions', 'route'), ('admissions', 'admission_location'),
        ('transfers', 'careunit'), ('d_labitems', 'label'), ('microbiologyevents', 'spec_type_desc')
    ]
    
    value_mapping = db_manager.get_value_mapping_batch(tables_and_columns)
    
    # Special cases for d_items
    special_queries = [
        ('d_items', 'label', 'linksto = "chartevents"', 'vital_name'),
        ('d_items', 'label', 'linksto = "inputevents"', 'input_name'),
        ('d_items', 'label', 'linksto = "outputevents"', 'output_name')
    ]
    
    for table, column, condition, mapping_key in special_queries:
        try:
            with get_db_connection(PROCESSED_DB_PATH) as processed_conn, get_db_connection(UNPROCESSED_DB_PATH) as unprocessed_conn:
                sql = f'select {column} from {table} where {condition}'
                processed_df = pd.read_sql_query(sql, processed_conn)
                unprocessed_df = pd.read_sql_query(sql, unprocessed_conn)
                
                # Filter vital signs if needed
                if mapping_key == 'vital_name':
                    excluded_vitals = [
                        'Temperature Celsius', 'O2 saturation pulseoxymetry', 'Heart Rate', 
                        'Respiratory Rate', 'Arterial Blood Pressure systolic', 
                        'Arterial Blood Pressure diastolic', 'Arterial Blood Pressure mean'
                    ]
                    mask = ~unprocessed_df[column].isin(excluded_vitals)
                    processed_df = processed_df[mask]
                    unprocessed_df = unprocessed_df[mask]
                
                value_mapping[mapping_key] = db_manager._create_value_mapping(processed_df[column], unprocessed_df[column])
        except Exception as e:
            print(f"Warning: Failed to process {mapping_key}: {e}")
    
    # Add aliases
    aliases = {
        'procedure_name1': 'procedure_name', 'procedure_name2': 'procedure_name',
        'diagnosis_name1': 'diagnosis_name', 'diagnosis_name2': 'diagnosis_name',
        'drug_name1': 'drug_name', 'drug_name2': 'drug_name', 'drug_name3': 'drug_name'
    }
    for alias, original in aliases.items():
        value_mapping[alias] = value_mapping.get(original, {})
    
    # Abbreviation mapping
    try:
        with get_db_connection(PROCESSED_DB_PATH) as processed_conn, get_db_connection(UNPROCESSED_DB_PATH) as unprocessed_conn:
            sql = 'select long_title from d_icd_diagnoses union all select long_title from d_icd_procedures'
            processed_df = pd.read_sql_query(sql, processed_conn)
            unprocessed_df = pd.read_sql_query(sql, unprocessed_conn)
            value_mapping['abbreviation'] = db_manager._create_value_mapping(processed_df['long_title'], unprocessed_df['long_title'])
    except Exception as e:
        print(f"Warning: Failed to create abbreviation mapping: {e}")
    
    return value_mapping


def process_questions_efficiently(questions_df, id2sample, value_mapping):
    """Process questions using optimized classes and methods."""
    with open(VALUE_MAPPING_FILE, 'r') as f:
        value_mapping_natural = json.load(f)
    
    # Get valid mappings and process questions
    mapping_processor = ValueMappingProcessor(value_mapping, value_mapping_natural)
    value_mapping_valid = mapping_processor.get_valid_mappings(questions_df, id2sample)
    
    question_processor = QuestionProcessor(value_mapping_valid, value_mapping_natural)
    questions_df_raw = question_processor.process_questions_batch(questions_df, id2sample)
    
    # Generate question context
    assumption_generator = AssumptionGenerator()
    assumptions = assumption_generator.generate_assumptions(questions_df_raw, id2sample)
    questions_df_raw['assumption'] = assumptions

    print(f"Processed {len(questions_df_raw)} questions successfully")
    return questions_df_raw


def execute_query_chunk(chunk_df):
    """Worker function for executing a chunk of queries."""
    db_manager = DatabaseManager(PROCESSED_DB_PATH, UNPROCESSED_DB_PATH)
    results = []
    
    for _, row in chunk_df.iterrows():
        result = db_manager.execute_query_safe(row['query_fixed_raw'])
        results.append({
            'question_id': row['question_id'],
            'answer': str(result.values) if result is not None else None,
            'success': result is not None
        })
    
    return results


def execute_queries_and_get_answers(questions_df_raw, multicore=True):
    """Execute queries and collect answers efficiently."""
    if multicore:
        # Parallel processing
        num_workers = mp.cpu_count()
        chunk_size = max(1, len(questions_df_raw) // num_workers)
        chunks = [questions_df_raw.iloc[i:i + chunk_size] for i in range(0, len(questions_df_raw), chunk_size)]
        
        all_results = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(execute_query_chunk, chunk) for chunk in chunks]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Executing queries"):
                all_results.extend(future.result())
        
        # Build results mapping
        question_id_to_answer = {r['question_id']: r['answer'] for r in all_results if r['success']}
        failed_queries = sum(1 for r in all_results if not r['success'])
        
        # Create final dataframe
        successful_rows = []
        for _, row in questions_df_raw.iterrows():
            if row['question_id'] in question_id_to_answer:
                row_dict = row.to_dict()
                row_dict['question_fixed_raw_answer'] = question_id_to_answer[row['question_id']]
                successful_rows.append(row_dict)
        
        final_df = pd.DataFrame(successful_rows)
    else:
        # Single-threaded processing
        db_manager = DatabaseManager(PROCESSED_DB_PATH, UNPROCESSED_DB_PATH)
        successful_rows = []
        failed_queries = 0
        
        for _, row in tqdm(questions_df_raw.iterrows(), total=len(questions_df_raw), desc="Executing queries"):
            result = db_manager.execute_query_safe(row['query_fixed_raw'])
            if result is not None:
                row_dict = row.to_dict()
                row_dict['question_fixed_raw_answer'] = str(result.values)
                successful_rows.append(row_dict)
            else:
                failed_queries += 1
        
        final_df = pd.DataFrame(successful_rows)
    
    print(f"Failed queries: {failed_queries}, Successful queries: {len(final_df)}")
    return final_df


def main(multicore=True):
    """Main execution function."""
    print("Starting data creation pipeline...")
    print(f"Multiprocessing: {'enabled' if multicore else 'disabled'}")
    
    # Setup and process
    db, raw_db = setup_databases()
    questions_df = load_and_prepare_questions()
    patients_df = build_patient_dataframe()
    
    # Merge questions with patient data
    questions_df = questions_df.merge(
        patients_df[["subject_id", "anchor_year", "fhir_id"]],
        on="subject_id", how="inner"
    )
    questions_df = apply_anchor_replacements(questions_df)
    
    # Process questions
    id2sample = create_sample_mapping(DATA_PATHS)
    value_mapping = build_value_mappings()
    questions_df_raw = process_questions_efficiently(questions_df, id2sample, value_mapping)
    final_df = execute_queries_and_get_answers(questions_df_raw, multicore)
    
    # Save results
    final_df_publish = final_df[[
        "split", "question_id", "question_fixed_raw", "query_fixed_raw", 
        "question_fixed_raw_answer", "assumption", "fhir_id", "template", "val_dict_fixed"
    ]].copy()
    
    final_df_publish.rename(columns={
        "val_dict_fixed": "val_dict", "question_fixed_raw": "question",
        "query_fixed_raw": "sql_query", "question_fixed_raw_answer": "true_answer",
        "fhir_id": "patient_fhir_id"
    }, inplace=True)
    
    os.makedirs(os.path.dirname(QUESTION_ANSWER_OUTPUT_CSV), exist_ok=True)
    final_df_publish.to_csv(QUESTION_ANSWER_OUTPUT_CSV, index=False)
    
    print(f"Results saved to {QUESTION_ANSWER_OUTPUT_CSV}")
    print(f"Final dataset shape: {final_df.shape}")
    print("Pipeline completed successfully!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Create question-answer dataset')
    parser.add_argument('--multicore', action='store_true', default=True, help='Enable multiprocessing')
    args = parser.parse_args()
    main(args.multicore)
