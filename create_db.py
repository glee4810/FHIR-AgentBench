import sys
import shutil
import pandas as pd
import os
import sqlite3
from typing import Optional, Dict, List, Union
from pathlib import Path

def read_csv(data_dir: str, 
            filename: str, 
            columns: Optional[List[str]] = None,
            lower: bool = False,
            filter_dict: Optional[Dict] = None, 
            dtype: Optional[Dict] = None,
            memory_efficient: bool = False) -> pd.DataFrame:
    """
    Read CSV file with optimized memory usage and error handling
    """
    filepath = Path(data_dir) / filename
    gz_filepath = filepath.with_suffix('.csv.gz')

    if not filepath.exists():
        print(f"File {filepath} does not exist")
        print("Trying to read as gzip")
        if not gz_filepath.exists():
            print(f"File {gz_filepath} does not exist")
            sys.exit(1)
        filepath = gz_filepath

    if memory_efficient:
        import dask.dataframe as dd
        from dask.diagnostics import ProgressBar

        ProgressBar().register()
        compression = "gzip" if str(filepath).endswith("gz") else None
        
        df = dd.read_csv(filepath, 
                        blocksize=25e6,
                        dtype=dtype,
                        compression=compression)
        
        if columns is not None:
            df = df[columns]
        if filter_dict:
            df = df[pd.concat([df[key].isin(val) for key, val in filter_dict.items()]).all(axis=0)]
        df = df.compute()
    else:
        try:
            df = pd.read_csv(filepath, usecols=columns, dtype=dtype)
        except:  # demo dbs have lowercased column names
            df = pd.read_csv(filepath, 
                           usecols=[col.lower() for col in columns] if columns else None,
                           dtype=dtype)
            df.columns = df.columns.str.upper()
            
        if filter_dict:
            df = df[pd.concat([df[key].isin(val) for key, val in filter_dict.items()]).all(axis=0)]

    if lower:
        str_cols = df.select_dtypes(include=['object']).columns
        df[str_cols] = df[str_cols].apply(lambda x: x.str.lower().str.strip())
    return df

class unprocessed_db:
    def __init__(self, 
                 db_name: str = "mimic_iv_unprocessed",
                 generate: bool = False,
                 out_dir: str = "data",
                 lower: bool = False):
        
        self.out_dir = Path(out_dir)
        self.db_path = self.out_dir / f"{db_name}.sqlite"
            
        if generate:
            if self.db_path.exists():
                print(f"File {self.db_path} already exists, deleting pre-existing one")
                self.db_path.unlink()
            self.generate_db(db_name=db_name, lower=lower)
            return
            
        self.conn = sqlite3.connect(self.db_path)
        self.cur = self.conn.cursor()

    def generate_db(self, db_name: str, lower: bool) -> None:
        self.conn = sqlite3.connect(self.db_path)
        self.cur = self.conn.cursor()
        
        table_names = [
            "patients", "admissions", "d_icd_diagnoses", "d_icd_procedures",
            "d_items", "d_labitems", "diagnoses_icd", "procedures_icd",
            "labevents", "prescriptions", "chartevents", "inputevents",
            "outputevents", "microbiologyevents", "icustays", "transfers"
        ]

        mimic_dtypes = {
            "subject_id": pd.Int64Dtype(),
            "hadm_id": pd.Int64Dtype(),
            "stay_id": pd.Int64Dtype(),
            "caregiver_id": pd.Int64Dtype(),
            "provider_id": str,
            "category": str,
            "parent_field_ordinal": str,
            "pharmacy_id": pd.Int64Dtype(),
            "emar_seq": pd.Int64Dtype(),
            "poe_seq": pd.Int64Dtype(),
            "ndc": str,
            "doses_per_24_hrs": pd.Int64Dtype(),
            "drg_code": str,
            "org_itemid": pd.Int64Dtype(),
            "isolate_num": pd.Int64Dtype(),
            "quantity": str,
            "ab_itemid": pd.Int64Dtype(),
            "dilution_text": str,
            "warning": pd.Int64Dtype(),
            "valuenum": float,
        }

        adm_df = None
        for table in table_names:
            rows = read_csv(self.out_dir, f"{table}.csv", lower=lower, dtype=mimic_dtypes)
            rows["row_id"] = f"{table}_" + rows.index.astype(str)
            rows.insert(0, "row_id", rows.pop("row_id"))
            
            if table == "procedures_icd":
                rows["charttime"] = pd.to_datetime(rows["chartdate"]).dt.strftime("%Y-%m-%d %H:%M:%S")
            elif table == "diagnoses_icd":
                if adm_df is None:
                    adm_df = read_csv(self.out_dir, "admissions.csv", 
                                    columns=["hadm_id", "admittime"],
                                    lower=lower, dtype=mimic_dtypes)
                rows = rows.merge(adm_df, on="hadm_id", how="left")
                rows.rename(columns={"admittime": "charttime"}, inplace=True)
                
            rows.to_sql(table, self.conn, if_exists="append", index=False)

    def quick_check(self) -> pd.Series:
        return pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", 
                               self.conn)["name"]

    def run_query(self, query: str) -> pd.DataFrame:
        return pd.read_sql_query(query, self.conn)
    
    def run_sql_query_local_sqlite(self, query: str) -> Optional[Union[pd.DataFrame, str]]:
        if not query or query.lower() == "null":
            return None
        try:
            return self.run_query(query)
        except Exception as e:
            print(f"Error executing query: {e}")
            return f"ERROR: {e}"