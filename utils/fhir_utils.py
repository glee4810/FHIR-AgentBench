import pandas as pd
import sqlite3
import json
import re
import copy
from contextlib import contextmanager
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm
import logging

from config import (
    TEXT_COLUMNS, 
    VITAL_SIGNS_MAPPING, FREQUENT_TERM_MAPPER, LABEL_MAPPER,
    VITAL_TEMPLATES, WEIGHT_TEMPLATES, HEIGHT_TEMPLATES, 
    GENDER_TEMPLATES, ER_TEMPLATES, VALUE_REPLACEMENT_PATTERNS,
    SUM_TEMPLATES, DOSE_VAL_RX_TEMPLATES
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@contextmanager
def get_db_connection(db_path: str):
    """Context manager for database connections."""
    conn = sqlite3.connect(db_path)
    try:
        yield conn
    finally:
        conn.close()


class DatabaseManager:
    """Manages database operations efficiently."""
    
    def __init__(self, processed_db_path: str, unprocessed_db_path: str):
        self.processed_db_path = processed_db_path
        self.unprocessed_db_path = unprocessed_db_path
    
    def get_value_mapping_batch(self, tables_and_columns: List[Tuple[str, str]]) -> Dict[str, Dict[str, List[str]]]:
        """Get value mappings for multiple tables/columns in batch."""
        value_mapping = {}
        
        with get_db_connection(self.processed_db_path) as processed_conn:
            with get_db_connection(self.unprocessed_db_path) as unprocessed_conn:
                
                for table, column in tables_and_columns:
                    sql = f'select {column} from {table}'
                    try:
                        processed_df = pd.read_sql_query(sql, processed_conn)
                        unprocessed_df = pd.read_sql_query(sql, unprocessed_conn)
                        
                        mapping_key = self._get_mapping_key(table, column)
                        value_mapping[mapping_key] = self._create_value_mapping(
                            processed_df[column], unprocessed_df[column]
                        )
                    except Exception as e:
                        logger.warning(f"Failed to process {table}.{column}: {e}")
        
        return value_mapping
    
    def _get_mapping_key(self, table: str, column: str) -> str:
        """Generate mapping key based on table and column."""
        mapping_keys = {
            ('patients', 'gender'): 'gender',
            ('d_icd_procedures', 'long_title'): 'procedure_name',
            ('d_icd_diagnoses', 'long_title'): 'diagnosis_name',
            ('prescriptions', 'drug'): 'drug_name',
            ('prescriptions', 'route'): 'drug_route',
            ('admissions', 'admission_location'): 'admission_route',
            ('transfers', 'careunit'): 'careunit',
            ('d_labitems', 'label'): 'lab_name',
            ('microbiologyevents', 'spec_type_desc'): 'spec_name'
        }
        return mapping_keys.get((table, column), f"{table}_{column}")
    
    def _create_value_mapping(self, processed_series: pd.Series, unprocessed_series: pd.Series) -> Dict[str, List[str]]:
        """Create value mapping between processed and unprocessed data."""
        mapping = {}
        for processed_val, unprocessed_val in zip(processed_series, unprocessed_series):
            if processed_val not in mapping:
                mapping[processed_val] = []
            if unprocessed_val not in mapping[processed_val]:
                mapping[processed_val].append(unprocessed_val)
        return mapping
    
    def execute_query_safe(self, query: str) -> Optional[pd.DataFrame]:
        """Execute SQL query safely with error handling."""
        try:
            with get_db_connection(self.unprocessed_db_path) as conn:
                return pd.read_sql_query(query, conn)
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return None


class ValueMappingProcessor:
    """Handles value mapping operations efficiently."""
    
    def __init__(self, value_mapping: Dict[str, Dict[str, List[str]]], 
                 value_mapping_natural: Dict[str, Dict[str, str]]):
        self.value_mapping = value_mapping
        self.value_mapping_natural = value_mapping_natural
    
    def get_valid_mappings(self, questions_df: pd.DataFrame, id2sample: Dict[str, Any]) -> Dict[str, Dict[str, List[str]]]:
        """Extract only valid value mappings used in questions."""
        value_mapping_valid = {}
        
        for _, row in tqdm(questions_df.iterrows(), total=len(questions_df), desc="Processing value mappings"):
            question_id = row['question_id']
            val_dict = id2sample[question_id]['val_dict']
            
            if 'val_placeholder' not in val_dict:
                continue
                
            for entity, pre_value in val_dict['val_placeholder'].items():
                if entity in self.value_mapping and pre_value in self.value_mapping[entity]:
                    post_values = self.value_mapping[entity][pre_value]
                    
                    if entity not in value_mapping_valid:
                        value_mapping_valid[entity] = {}
                    if pre_value not in value_mapping_valid[entity]:
                        value_mapping_valid[entity][pre_value] = post_values
        
        return value_mapping_valid


class QuestionProcessor:
    """Processes questions and queries efficiently."""
    
    def __init__(self, value_mapping_valid: Dict[str, Dict[str, List[str]]], 
                 value_mapping_natural: Dict[str, Dict[str, str]]):
        self.value_mapping_valid = value_mapping_valid
        self.value_mapping_natural = value_mapping_natural
        # Load question fixes from file
        try:
            with open('question_fixes_complete.json', 'r') as f:
                self.question_fixes = json.load(f)
        except FileNotFoundError:
            self.question_fixes = {}
    
    def process_questions_batch(self, questions_df: pd.DataFrame, id2sample: Dict[str, Any]) -> pd.DataFrame:
        """Process questions in batch for better performance."""
        processed_questions = []
        
        # First pass: handle questions without multiple post values
        for _, row in tqdm(questions_df.iterrows(), total=len(questions_df), desc="Processing single-value questions"):
            if not self._has_multiple_post_values(row, id2sample):
                processed_row = self._process_single_question(row, id2sample, False)
                if processed_row is not None:
                    processed_questions.append(processed_row)
        
        # Second pass: handle questions with multiple post values (like original code)
        for _, row in tqdm(questions_df.iterrows(), total=len(questions_df), desc="Processing multi-value questions"):
            if self._has_multiple_post_values(row, id2sample):
                processed_row = self._process_multiple_value_question(row, id2sample)
                if processed_row is not None:
                    processed_questions.append(processed_row)
        
        return pd.DataFrame(processed_questions) if processed_questions else pd.DataFrame()
    
    def _has_multiple_post_values(self, row: pd.Series, id2sample: Dict[str, Any]) -> bool:
        """Check if any entity has multiple post values."""
        question_id = row['question_id']
        val_dict = id2sample[question_id]['val_dict']
        
        if 'val_placeholder' not in val_dict:
            return False
            
        for entity, pre_value in val_dict['val_placeholder'].items():
            if (entity in self.value_mapping_valid and 
                pre_value in self.value_mapping_valid[entity] and
                len(self.value_mapping_valid[entity][pre_value]) > 1):
                return True
        return False
    
    def _process_multiple_value_question(self, row: pd.Series, id2sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process questions with multiple post values (replicates original logic)."""
        question_id = row['question_id']
        val_dict = id2sample[question_id]['val_dict']
        template = id2sample[question_id]['template']
        
        question_fixed = row['question_fixed']
        query_fixed = row['query_fixed']
        
        # Apply hard-coded fixes first if question ID is in the fixes
        if question_id in self.question_fixes:
            query_fixed = self.question_fixes[question_id]
        
        # Apply value replacements to question text (not query, since query is fixed)
        if 'val_placeholder' in val_dict:
            for entity, pre_value in val_dict['val_placeholder'].items():
                if (entity in self.value_mapping_valid and 
                    pre_value in self.value_mapping_valid[entity]):
                    
                    post_value_question = self.value_mapping_natural[entity][pre_value]
                    
                    # Apply pattern replacements to question text
                    question_fixed = self._replace_value_patterns_question_only(
                        question_fixed, pre_value, post_value_question
                    )
        
        # Apply template-specific fixes
        query_fixed = self._apply_template_fixes(query_fixed, template)
        
        result = row.to_dict()
        result['question_fixed_raw'] = question_fixed.capitalize()
        result['query_fixed_raw'] = query_fixed
        
        return result
    
    def _replace_value_patterns_question_only(self, question: str, pre_value: str, post_value_question: str) -> str:
        """Replace value patterns in question text only."""
        patterns = []
        for old_pattern, new_pattern in VALUE_REPLACEMENT_PATTERNS:
            formatted_old = old_pattern.format(
                pre_value=pre_value,
                post_value=post_value_question,
                pre_value_capitalized=pre_value.capitalize(),
                post_value_capitalized=post_value_question.capitalize()
            )
            formatted_new = new_pattern.format(
                pre_value=pre_value,
                post_value=post_value_question,
                pre_value_capitalized=pre_value.capitalize(),
                post_value_capitalized=post_value_question.capitalize()
            )
            patterns.append((formatted_old, formatted_new))
        
        for old_pattern, new_pattern in patterns:
            if question.count(old_pattern) == 1:
                question = question.replace(old_pattern, new_pattern, 1)
                break
        
        return question
    
    def _process_single_question(self, row: pd.Series, id2sample: Dict[str, Any], is_multiple_value: bool = False) -> Optional[Dict[str, Any]]:
        """Process a single question with optimized value replacement."""
        question_id = row['question_id']
        val_dict = id2sample[question_id]['val_dict']
        template = id2sample[question_id]['template']
        
        question_fixed = row['question_fixed']
        query_fixed = row['query_fixed']
        
        # Apply value replacements
        question_fixed, query_fixed = self._apply_value_replacements(
            question_fixed, query_fixed, val_dict
        )
        
        # Apply template-specific fixes
        query_fixed = self._apply_template_fixes(query_fixed, template)
        
        result = row.to_dict()
        result['question_fixed_raw'] = question_fixed.capitalize()
        result['query_fixed_raw'] = query_fixed
        
        return result
    
    def _apply_value_replacements(self, question: str, query: str, val_dict: Dict[str, Any]) -> Tuple[str, str]:
        """Apply value replacements efficiently."""
        if 'val_placeholder' not in val_dict:
            return question, query
            
        for entity, pre_value in val_dict['val_placeholder'].items():
            if (entity in self.value_mapping_valid and 
                pre_value in self.value_mapping_valid[entity]):
                
                post_value_question = self.value_mapping_natural[entity][pre_value]
                post_value_sql = self.value_mapping_valid[entity][pre_value][0]
                
                # Apply replacements using vectorized operations
                question, query = self._replace_value_patterns(
                    question, query, pre_value, post_value_question, post_value_sql
                )
        
        return question, query
    
    def _replace_value_patterns(self, question: str, query: str, pre_value: str, 
                               post_value_question: str, post_value_sql: str) -> Tuple[str, str]:
        """Replace value patterns efficiently."""
        patterns = []
        for old_pattern, new_pattern in VALUE_REPLACEMENT_PATTERNS:
            formatted_old = old_pattern.format(
                pre_value=pre_value,
                post_value=post_value_question,
                pre_value_capitalized=pre_value.capitalize(),
                post_value_capitalized=post_value_question.capitalize()
            )
            formatted_new = new_pattern.format(
                pre_value=pre_value,
                post_value=post_value_question,
                pre_value_capitalized=pre_value.capitalize(),
                post_value_capitalized=post_value_question.capitalize()
            )
            patterns.append((formatted_old, formatted_new))
        
        for old_pattern, new_pattern in patterns:
            if question.count(old_pattern) == 1:
                question = question.replace(old_pattern, new_pattern, 1)
                query = query.replace(f"'{pre_value}'", f"'{post_value_sql}'")
                break
        
        return question, query
    
    def _apply_template_fixes(self, query: str, template: str) -> str:
        """Apply template-specific fixes."""
        # Vital signs fixes
        if template in VITAL_TEMPLATES:
            query = self._apply_vital_signs_fixes(query)
        
        # Weight fixes
        elif template in WEIGHT_TEMPLATES:
            query = query.replace("daily weight", "Daily Weight")
        
        # Height fixes
        elif template in HEIGHT_TEMPLATES:
            query = query.replace("height (cm)", "Height (cm)")
        
        # Gender fixes
        elif template in GENDER_TEMPLATES:
            query = self._apply_gender_fixes(query)
        
        # Emergency room fixes
        elif template in ER_TEMPLATES:
            query = query.replace("emergency room", "EMERGENCY ROOM")
        
        return query
    
    def _apply_vital_signs_fixes(self, query: str) -> str:
        """Apply vital signs specific fixes."""
        for vital_sign, mapping in VITAL_SIGNS_MAPPING.items():
            if f"'{vital_sign.lower()}'" in query:
                query = query.replace(vital_sign.lower(), vital_sign)
        return query
    
    def _apply_gender_fixes(self, query: str) -> str:
        """Apply gender-specific fixes."""
        if "'m'" in query:
            return query.replace("'m'", "'M'")
        elif "'f'" in query:
            return query.replace("'f'", "'F'")
        return query


class AssumptionGenerator:
    """Generates assumptions for questions efficiently."""
    
    def generate_assumptions(self, questions_df: pd.DataFrame, id2sample: Dict[str, Any]) -> List[str]:
        """Generate assumptions for all questions."""
        assumptions = []
        
        for _, row in questions_df.iterrows():
            assumption = self._generate_single_assumption(row, id2sample)
            assumptions.append(assumption)
        
        return assumptions
    
    def _generate_single_assumption(self, row: pd.Series, id2sample: Dict[str, Any]) -> str:
        """Generate assumption for a single question."""
        question_id = row['question_id']
        query_fixed_raw = row['query_fixed_raw']
        anchor_year = row['anchor_year']
        template = row['template']
        
        val_placeholder = id2sample[question_id]['val_dict'].get('val_placeholder', {})
        value_list = [str(v) for v in val_placeholder.values()]
        
        assumption_parts = []
        
        # Time expression
        if str(anchor_year) in query_fixed_raw:
            assumption_parts.append(f'Assume the current time is {anchor_year}-12-31 23:59:00.')
        
        # Frequent terms
        for key, description in FREQUENT_TERM_MAPPER.items():
            if key in query_fixed_raw:
                assumption_parts.append(description)
        
        # Text columns variation
        if set(val_placeholder.keys()).intersection(TEXT_COLUMNS):
            assumption_parts.append(
                'When searching for values in the database, account for all variations in letter case and surrounding whitespace.'
            )
        
        # Label mapper assumptions
        for key, table in LABEL_MAPPER.items():
            if key in value_list:
                assumption_parts.append(f'When searching for {key}, use the {table} records.')

        # SELECT SUM(...) assumptions
        if template in SUM_TEMPLATES:
            assumption_parts.append('I want the final output to be the sum of the retrieved results.')
        
        # Drug dose assumptions
        if template in DOSE_VAL_RX_TEMPLATES:
            assumption_parts.append('I want to use the dose_val_rx column for the dose of any medication.')
        
        return ' '.join(assumption_parts)


def load_data_efficiently(data_paths: List[str]) -> pd.DataFrame:
    """Load and combine data from multiple paths efficiently."""
    dataframes = []
    
    for file_path in data_paths:
        # Read annotated data
        with open(f"{file_path}/annotated.json", "r") as f:
            ann_data = json.load(f)
        
        ann_df = pd.DataFrame(ann_data)
        ann_df['split'] = file_path.split('/')[-1]
        ann_df.insert(0, 'split', ann_df.pop('split'))
        
        # Read answer data
        with open(f"{file_path}/answer.json", "r") as f:
            ans_data = json.load(f)
        
        ans_df = pd.DataFrame.from_dict(ans_data, orient="index").reset_index()
        ans_df.columns = ["id", "answer"]
        
        # Process answers efficiently
        ans_df["answer"] = ans_df["answer"].apply(
            lambda x: eval(x) if x != "null" else x
        )
        ans_df["num_answers"] = ans_df["answer"].apply(
            lambda x: len(x) if isinstance(x, list) else x
        )
        
        # Merge and append
        curr_df = pd.merge(ann_df, ans_df, on='id', how='inner')
        dataframes.append(curr_df)
    
    # Concatenate all dataframes at once
    full_df = pd.concat(dataframes, ignore_index=True)
    full_df.rename(columns={"id": "question_id"}, inplace=True)
    full_df = full_df[full_df["query"] != "null"]
    full_df.index = range(len(full_df))
    
    return full_df


def apply_anchor_replacements(questions_df: pd.DataFrame) -> pd.DataFrame:
    """Apply anchor year replacements to all rows at once (loop-free)."""
    questions_df = questions_df.copy()
    
    # Bulk anchor replacement (no explicit loop)
    def replace_anchor_serieswise(text_series, anchor_series):
        def replace_text_or_dict(text, anchor):
            if isinstance(text, dict):
                # Convert dict to string, do replacement, then back to dict
                text_str = str(text)
                replaced_str = re.sub(r"(?<!\d)2100(?!\d)", str(anchor), text_str)
                return eval(replaced_str)
            else:
                # Handle as string
                return re.sub(r"(?<!\d)2100(?!\d)", str(anchor), str(text))
        
        return text_series.combine(
            anchor_series, 
            replace_text_or_dict
        )

    questions_df["val_dict_fixed"] = replace_anchor_serieswise(
        questions_df["val_dict"], questions_df["anchor_year"]
    )

    questions_df["question_fixed"] = replace_anchor_serieswise(
        questions_df["question"], questions_df["anchor_year"]
    )
    
    questions_df["query_fixed"] = replace_anchor_serieswise(
        questions_df["query"], questions_df["anchor_year"]
    )
    
    # Post-process SQL for all rows at once
    def post_process_sql_serieswise(sql_series, anchor_series):
        def replace_current_time(sql, anchor):
            if isinstance(sql, dict):
                sql_str = str(sql)
                replaced_str = sql_str.replace("current_time", f"'{anchor}-12-31 00:00:00'")
                return eval(replaced_str)
            else:
                return sql.replace("current_time", f"'{anchor}-12-31 00:00:00'")
        
        return sql_series.combine(
            anchor_series, 
            replace_current_time
        )
    
    questions_df["val_dict_fixed"] = post_process_sql_serieswise(
        questions_df["val_dict_fixed"], questions_df["anchor_year"]
    )

    questions_df["query_fixed"] = post_process_sql_serieswise(
        questions_df["query_fixed"], questions_df["anchor_year"]
    )

    # Since-time modification
    def replace_since_time_filter(query_fixed, val_dict, anchor):
        if 'time_placeholder' in val_dict and 'time_filter_global1' in val_dict['time_placeholder'] and 'since' in val_dict['time_placeholder']['time_filter_global1']['type']:
            current_sql = val_dict['time_placeholder']['time_filter_global1']['sql']
            time_column = val_dict['time_placeholder']['time_filter_global1']['col']
            current_sql_list = []
            if isinstance(current_sql, list):
                for sql, col in zip(current_sql, time_column):
                    assert str(col) in str(sql), f"Column {col} not found in SQL: {sql}"
                    new_sql = sql + f" AND strftime('%Y-%m-%d',{col}) <= '{anchor}-12-31 23:59:00'"
                    current_sql_list.append(new_sql)
                    query_fixed = query_fixed.replace(sql, new_sql)
                val_dict['time_placeholder']['time_filter_global1']['sql'] = current_sql_list
            else:
                assert str(time_column[0]) in str(current_sql), f"Column {time_column} not found in SQL: {current_sql}"
                new_sql = current_sql + f" AND strftime('%Y-%m-%d',{time_column[0]}) <= '{anchor}-12-31 23:59:00'"
                query_fixed = query_fixed.replace(current_sql, new_sql)
                val_dict['time_placeholder']['time_filter_global1']['sql'] = new_sql
        return query_fixed, val_dict
    
    def apply_replace_since_time_filter(row):
        return replace_since_time_filter(row["query_fixed"], row["val_dict_fixed"], row["anchor_year"])
    
    questions_df[["query_fixed", "val_dict_fixed"]] = questions_df.apply(apply_replace_since_time_filter, axis=1, result_type='expand')
    return questions_df
