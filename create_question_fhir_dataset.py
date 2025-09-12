import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import random
import requests

from config import (
    QUESTION_ANSWER_OUTPUT_CSV,
    QUESTION_ANSWER_FHIR_OUTPUT_CSV,
    QUESTION_SQL_FHIR_OUTPUT_CSV
)
from fhir_client import get_fhir_client
from utils.sql_to_fhir_utils import (
    TABLE_TO_RESOURCE_MAP,
    RESOURCE_TO_GCP_FHIR_MAP,
    create_unique_table_id,
    convert_table_id_to_fhir_id,
    replace_sql_query,
    extract_first_subquery,
    extract_outer_subselects,
    assign_medmix_ids
)
from create_question_answer_dataset import setup_databases
from create_db import unprocessed_db


def get_raw_db():
    """Create a raw_db instance without generating."""
    return unprocessed_db(out_dir="data", db_name="mimic_iv_unprocessed_raw", generate=False, lower=False)


def load_and_preprocess_data(question_answer_output_csv):
    """Load and preprocess SQL answers with FHIR mapping info."""
    final_df_main = pd.read_csv(question_answer_output_csv)
    final_df_main["proc_query"] = final_df_main["sql_query"]
    print(f"Loaded final_df for FHIR conversion: {final_df_main.shape}")

    final_df_main["main_table_name"] = final_df_main["proc_query"].apply(
        lambda x: x.split("FROM ")[1].split(" ")[0]
    )
    final_df = final_df_main.copy()
    final_df["mappable_to_fhir"] = final_df["main_table_name"].isin(TABLE_TO_RESOURCE_MAP.keys())
    final_df = final_df[final_df["mappable_to_fhir"]==True]
    
    print(f"All main table names: \n{final_df['main_table_name'].value_counts(dropna=False)}")
    print(f"Mappable to FHIR: \n{final_df.loc[final_df['mappable_to_fhir'], 'main_table_name'].value_counts(dropna=False)}")
    
    return final_df, final_df_main


def process_row_to_fhir(row, updated_query, main_table_name, fhir_resource_type):
    """Execute SQL query and convert results to FHIR IDs."""
    raw_db_local = get_raw_db()
    query_result = raw_db_local.run_sql_query_local_sqlite(updated_query)
    
    if query_result is None or 'row_id' not in query_result.columns:
        return None
    
    # Add metadata
    query_result.insert(0, "question_id", row["question_id"])
    query_result["main_table_name"] = main_table_name
    query_result["fhir_resource"] = fhir_resource_type
    query_result["gcp_fhir_resource"] = RESOURCE_TO_GCP_FHIR_MAP[fhir_resource_type]
    
    if not query_result.empty:
        query_result["row_id"] = query_result["row_id"].values
        query_result["table_id"] = query_result.apply(
            lambda r: create_unique_table_id(r, main_table_name, fhir_resource_type), axis=1
        )
        query_result["fhir_id"] = query_result.apply(
            lambda r: convert_table_id_to_fhir_id(r["fhir_resource"], r["table_id"]), axis=1
        )
        
        # Handle MedicationPrescriptions special case
        if main_table_name == 'prescriptions' and fhir_resource_type == 'MedicationPrescriptions':
            medmix_df = query_result.copy()
            medmix_df["fhir_resource"] = "MedicationMix"
            medmix_df["gcp_fhir_resource"] = "Medication"
            medmix_df = assign_medmix_ids(medmix_df, resource_type='MedicationPrescriptions')
            query_result = pd.concat([query_result, medmix_df], ignore_index=True)
    else:
        query_result = pd.DataFrame({
            "question_id": [row["question_id"]],
            "row_id": [np.nan],
            "main_table_name": [main_table_name],
            "table_id": [np.nan],
            "fhir_resource": [np.nan],
            "gcp_fhir_resource": [np.nan],
            "fhir_id": [np.nan]
        })
    
    keep_cols = ["question_id", "row_id", "main_table_name", "table_id", "fhir_resource", "gcp_fhir_resource", "fhir_id"]
    return query_result.loc[:, query_result.columns.isin(keep_cols)].copy()


def process_row_primary(row):
    """Worker function for primary query processing."""
    updated_query = replace_sql_query(row["proc_query"])
    extracted_main_table_name = updated_query.split("FROM ")[1].split(" ")[0]
    
    main_table_name = (
        extracted_main_table_name
        if any(x in extracted_main_table_name for x in ["diagnoses", "procedures"])
        else row["main_table_name"]
    )
    
    if main_table_name not in TABLE_TO_RESOURCE_MAP.keys():
        return None

    fhir_resource_type = TABLE_TO_RESOURCE_MAP[main_table_name]
    results = []
    
    resource_types = fhir_resource_type if isinstance(fhir_resource_type, list) else [fhir_resource_type]
    for resource in resource_types:
        processed = process_row_to_fhir(row, updated_query, main_table_name, resource)
        if processed is not None:
            results.append(processed)
    
    return pd.concat(results, ignore_index=True) if results else None


def translate_sql_to_fhir(final_df_main, multicore=True):
    """Translate SQL query results to FHIR IDs for primary queries."""
    final_df = final_df_main[final_df_main["main_table_name"].isin(TABLE_TO_RESOURCE_MAP.keys())].copy()
    print(f"Filtered final_df shape for FHIR conversion: {final_df.shape}")
    print("Translating primary SQL query results to FHIR IDs...")

    translated_df_primary = pd.DataFrame()
    
    if multicore:
        with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
            futures = [executor.submit(process_row_primary, row) for _, row in final_df.iterrows()]
            for future in tqdm(as_completed(futures), total=len(futures)):
                processed = future.result()
                if processed is not None:
                    translated_df_primary = pd.concat([translated_df_primary, processed], ignore_index=True)
    else:
        for _, row in tqdm(final_df.iterrows(), total=len(final_df)):
            processed = process_row_primary(row)
            if processed is not None:
                translated_df_primary = pd.concat([translated_df_primary, processed], ignore_index=True)

    translated_df_primary.drop_duplicates(subset=['fhir_id', 'question_id'], inplace=True)
    print(f"Translated primary queries shape: {translated_df_primary.shape}")
    
    return translated_df_primary, final_df_main


def validate_fhir_ids(translated_df):
    """Validate generated FHIR IDs against the GCP FHIR store."""
    print("Validating FHIR IDs against GCP FHIR store...")
    fhir_client = get_fhir_client()
    
    def fetch_existing_ids(resource, ids):
        existing = set()
        chunk_size = 50
        for i in tqdm(range(0, len(ids), chunk_size), desc=f"Processing {resource}"):
            chunk = ids[i:i + chunk_size]
            attempt = 0
            while attempt < 3:
                try:
                    result = fhir_client.get_resources_by_resource_ids(resource, chunk)
                    existing.update(res['id'] for res in result if 'id' in res)
                    break
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 429:
                        sleep_time = (2 ** attempt) + random.uniform(0, 1)
                        print(f"Rate limit hit. Retrying after {sleep_time:.2f}s...")
                        time.sleep(sleep_time)
                        attempt += 1
                    else:
                        print(f"Error fetching {resource}: {e}")
                        break
            time.sleep(0.1)
        return existing
    
    existing_per_resource = {}
    for resource, group in translated_df.groupby("gcp_fhir_resource"):
        ids = group["fhir_id"].dropna().unique().tolist()
        if ids:
            existing_per_resource[resource] = fetch_existing_ids(resource, ids)
    
    def check_row(row):
        if pd.notnull(row["fhir_id"]):
            existing = existing_per_resource.get(row["gcp_fhir_resource"], set())
            return 1 if row["fhir_id"] in existing else 0
        return None
    
    translated_df["check_fhir_id"] = translated_df.apply(check_row, axis=1)
    print("FHIR ID validation results:")
    print(translated_df["check_fhir_id"].value_counts(dropna=False))
    
    return translated_df


def filter_invalid_fhir_ids(translated_df):
    """Filter out invalid FHIR IDs based on validation results."""
    print(f"Shape before filtering: {translated_df.shape}")
    
    microbio_mask = translated_df["main_table_name"] == "microbiologyevents"
    non_microbio = translated_df[~microbio_mask].copy()
    
    # Remove questions with any invalid FHIR IDs (except microbio)
    bad_question_ids = non_microbio.loc[non_microbio["check_fhir_id"] == 0, "question_id"].unique()
    filtered_non_microbio = non_microbio[~non_microbio["question_id"].isin(bad_question_ids)].copy()
    
    # For microbio, only remove invalid rows
    microbio = translated_df[microbio_mask].copy()
    filtered_microbio = microbio[microbio["check_fhir_id"] != 0].copy()
    
    result = pd.concat([filtered_non_microbio, filtered_microbio], ignore_index=True)
    print(f"Shape after filtering: {result.shape}")
    
    return result


def process_row_nested(row):
    """Worker function for nested query processing."""
    updated_query = replace_sql_query(row["subquery"])
    main_table_name = updated_query.split("FROM ")[1].split(" ")[0]
    fhir_resource_type = TABLE_TO_RESOURCE_MAP[main_table_name]
    
    results = []
    resource_types = fhir_resource_type if isinstance(fhir_resource_type, list) else [fhir_resource_type]
    for resource in resource_types:
        processed = process_row_to_fhir(row, updated_query, main_table_name, resource)
        if processed is not None:
            results.append(processed)
    
    return pd.concat(results, ignore_index=True) if results else None


def process_nested_queries(final_df_main, multicore=True):
    """Process nested SQL queries for FHIR ID translation."""
    print("Processing nested queries for FHIR ID translation...")
    
    # Follow the original logic: first extract main_table_name, then filter by '('
    final_df_sql_nested = final_df_main.copy()
    final_df_sql_nested["main_table_name"] = final_df_sql_nested["proc_query"].apply(
        lambda x: x.split("FROM ")[1].split(" ")[0]
    )
    df_nested_queries = final_df_sql_nested[final_df_sql_nested["main_table_name"] == '('].copy()
    print(f"Number of nested queries: {df_nested_queries.shape[0]}")
    
    if df_nested_queries.empty:
        return pd.DataFrame()
    
    df_nested_queries["subquery"] = df_nested_queries["proc_query"].apply(extract_first_subquery)
    
    translated_df_subqueries = pd.DataFrame()
    
    if multicore:
        with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
            futures = [executor.submit(process_row_nested, row) for _, row in df_nested_queries.iterrows()]
            for future in tqdm(as_completed(futures), total=len(futures)):
                processed = future.result()
                if processed is not None:
                    translated_df_subqueries = pd.concat([translated_df_subqueries, processed], ignore_index=True)
    else:
        for _, row in tqdm(df_nested_queries.iterrows(), total=len(df_nested_queries)):
            processed = process_row_nested(row)
            if processed is not None:
                translated_df_subqueries = pd.concat([translated_df_subqueries, processed], ignore_index=True)

    translated_df_subqueries.drop_duplicates(subset=['fhir_id', 'question_id'], inplace=True)
    print(f"Translated nested queries shape: {translated_df_subqueries.shape}")
    
    return translated_df_subqueries


def process_row_special(row, subquery_cleaned):
    """Worker function for special template processing."""
    row_copy = row.copy()
    row_copy["proc_query"] = subquery_cleaned
    
    # Extract main table name more carefully
    try:
        from_part = subquery_cleaned.split("FROM ")[1]
        main_table_name = from_part.split(" ")[0]
        
        # Handle cases where table name might still contain parentheses or complex expressions
        if main_table_name.startswith('(') or main_table_name not in TABLE_TO_RESOURCE_MAP:
            # If we can't find a valid table name, skip this row
            return None
            
    except (IndexError, KeyError):
        # If parsing fails, skip this row
        return None
    
    fhir_resource_type = TABLE_TO_RESOURCE_MAP[main_table_name]
    row_copy["main_table_name"] = main_table_name
    row_copy["fhir_resource"] = fhir_resource_type
    
    return process_row_to_fhir(row_copy, subquery_cleaned, main_table_name, fhir_resource_type)


def process_special_template(question_answers_sql_current_csv, multicore=True):
    """Process special template (Intake/Output Difference) for FHIR ID translation."""
    print("Processing special template (Intake/Output Difference)...")
    
    final_df_sql_template_special = pd.read_csv(question_answers_sql_current_csv).rename(columns={'sql_query': 'proc_query'})
    select_template_special_io = ['What is the difference between the total volume of intake and output of patient {patient_id} [time_filter_global1]?']
    
    final_df_sql_template_special = final_df_sql_template_special[
        final_df_sql_template_special["template"].isin(select_template_special_io)
    ].copy()
    
    print(f"Special template questions shape: {final_df_sql_template_special.shape}")
    
    if final_df_sql_template_special.empty:
        return pd.DataFrame()
    
    translated_df_template_special_io = pd.DataFrame()
    tasks = []
    
    for _, row in final_df_sql_template_special.iterrows():
        subqueries = extract_outer_subselects(row["proc_query"])
        for subquery in subqueries:
            tasks.append((row, subquery[1:-1]))  # Remove parentheses
    
    if multicore:
        with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
            futures = [executor.submit(process_row_special, row, subquery_cleaned) for row, subquery_cleaned in tasks]
            for future in tqdm(as_completed(futures), total=len(futures)):
                processed = future.result()
                if processed is not None:
                    translated_df_template_special_io = pd.concat([translated_df_template_special_io, processed], ignore_index=True)
    else:
        for row, subquery_cleaned in tqdm(tasks, total=len(tasks)):
            processed = process_row_special(row, subquery_cleaned)
            if processed is not None:
                translated_df_template_special_io = pd.concat([translated_df_template_special_io, processed], ignore_index=True)

    translated_df_template_special_io.drop_duplicates(subset=['fhir_id', 'question_id'], inplace=True)
    print(f"Special template shape: {translated_df_template_special_io.shape}")
    
    return translated_df_template_special_io


def combine_sql_fhir_df(sql_df, fhir_df):
    """Combine SQL and FHIR dataframes."""
    print("SQL dataframe shape:", sql_df.shape)
    
    # FHIR DF, by question id level
    fhir_df_by_question = fhir_df.copy()
    fhir_df_by_question["fhir_id"] = fhir_df["fhir_id"].fillna("")
    fhir_df_by_question = (
        fhir_df_by_question
        .groupby("question_id")
        .apply(lambda x: x.groupby("gcp_fhir_resource")["fhir_id"].apply(lambda y: list(set(y))).to_dict())
        .reset_index(name="true_fhir_ids")
    )
    print("FHIR dataframe shape:", fhir_df_by_question.shape)
    
    # Merge dataframes
    cols = [col for col in fhir_df_by_question.columns if col not in sql_df.columns or col == "question_id"]
    ground_truth_df = sql_df.merge(fhir_df_by_question[cols], on="question_id", how="outer")
    ground_truth_df["true_fhir_ids"] = ground_truth_df["true_fhir_ids"].fillna(dict())
    
    print("Merged dataframe shape:", ground_truth_df.shape)
    return ground_truth_df


def main(multicore=True):
    """Main execution function."""
    print("Starting FHIR data creation pipeline...")
    print(f"Multiprocessing: {'enabled' if multicore else 'disabled'}")

    # Setup databases
    db, raw_db = setup_databases()
    db.quick_check()

    # Load and process data
    final_df, final_df_main = load_and_preprocess_data(QUESTION_ANSWER_OUTPUT_CSV)
    
    # Primary queries
    translated_df_primary, final_df_main = translate_sql_to_fhir(final_df, multicore)
    translated_df_primary = validate_fhir_ids(translated_df_primary)
    translated_df_primary = filter_invalid_fhir_ids(translated_df_primary)
    print(f"Unique question IDs after primary translation: {translated_df_primary.question_id.nunique()}")

    # Nested queries
    translated_df_subqueries = process_nested_queries(final_df_main, multicore)
    if not translated_df_subqueries.empty:
        translated_df_subqueries = validate_fhir_ids(translated_df_subqueries)
        translated_df_subqueries = filter_invalid_fhir_ids(translated_df_subqueries)

    # Special template
    translated_df_template_special_io = process_special_template(QUESTION_ANSWER_OUTPUT_CSV, multicore)
    if not translated_df_template_special_io.empty:
        translated_df_template_special_io = validate_fhir_ids(translated_df_template_special_io)
        translated_df_template_special_io = filter_invalid_fhir_ids(translated_df_template_special_io)

    # Combine all results
    final_translated_df = pd.concat([
        translated_df_primary,
        translated_df_subqueries,
        translated_df_template_special_io
    ], ignore_index=True)
    
    final_translated_df.drop_duplicates(subset=['fhir_id', 'question_id'], inplace=True)
    
    # Create ground truth file
    ground_truth_df = combine_sql_fhir_df(final_df, final_translated_df)
    
    # Save files
    final_translated_df.to_csv(QUESTION_ANSWER_FHIR_OUTPUT_CSV, index=False)
    ground_truth_df.to_csv(QUESTION_SQL_FHIR_OUTPUT_CSV, index=False)

    print(f"FHIR translated answers saved to {QUESTION_ANSWER_FHIR_OUTPUT_CSV}")
    print(f"SQL and FHIR merged dataframe saved to {QUESTION_SQL_FHIR_OUTPUT_CSV}")
    print(f"Final translated DataFrame shape: {final_translated_df.shape}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Convert SQL query results to FHIR IDs')
    parser.add_argument('--multicore', action='store_true', default=True, help='Enable multiprocessing')
    args = parser.parse_args()
    main(args.multicore)
