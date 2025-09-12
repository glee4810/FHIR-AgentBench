
import uuid
import re
import sqlglot
from sqlglot.expressions import Subquery
import sqlparse
import pandas as pd


TABLE_TO_RESOURCE_MAP = {
    "admissions": "Encounter",
    "icustays": "EncounterICU",
    "diagnoses_icd": "Condition",
    "d_icd_diagnoses": "Condition",
    "prescriptions": ["MedicationPrescriptions", "MedicationRequest"],
    "patients": "Patient",
    "procedures_icd": "Procedure",
    "d_icd_procedures": "Procedure",
    "chartevents": "ObservationChartevents",
    "inputevents": "MedicationAdministrationICU",
    "microbiologyevents": ["ObservationMicroTest", "ObservationMicroOrg", "ObservationMicroSusc", "Specimen"],
    "outputevents": "ObservationOutputevents",
    "transfers": "Location",
    "labevents": "ObservationLabevents"
}

RESOURCE_TO_GCP_FHIR_MAP = {
    "Encounter": "Encounter",
    "EncounterICU": "Encounter",
    "Condition": "Condition",
    "MedicationPrescriptions": "Medication",
    "MedicationRequest": "MedicationRequest",
    "Patient": "Patient",
    "Procedure": "Procedure",
    "ObservationChartevents": "Observation",
    "MedicationAdministrationICU": "MedicationAdministration",
    "ObservationMicroTest": "Observation",
    "ObservationMicroOrg": "Observation",
    "ObservationMicroSusc": "Observation",
    "ObservationOutputevents": "Observation",
    "Location": "Location",
    "ObservationLabevents": "Observation",
    "Specimen" : "Specimen"
}

def create_unique_table_id(row, table_name: str, resource_type = None):
    if table_name == "admissions":
        return str(row["hadm_id"])
    elif table_name == "icustays":
        return str(row["stay_id"])
    elif table_name == "diagnoses_icd":
        return "-".join(
            [
                str(row["hadm_id"]),
                str(row["seq_num"]),
                str(row["icd_code"]).upper(),
            ]
        )
    elif (table_name == "prescriptions") and (resource_type == "MedicationPrescriptions"):
        elem1 = row["drug"]
        elem2 = "--" + row["formulary_drug_cd"] if row["formulary_drug_cd"] is not None and row["formulary_drug_cd"] != "" else ""
        ndc_str = re.sub(r"\.[^.]*$", "", str(row["ndc"])) if row["ndc"] is not None else None
        # Need to add 0 padding to ndc_str if it's less than 11 digits
        elem3 = "--" + ndc_str.zfill(11) if ndc_str is not None and ndc_str != "0" and ndc_str != "" else ""
        return elem1 + elem2 + elem3
    elif (table_name == "prescriptions") and (resource_type == "MedicationRequest"):
        return str(row['pharmacy_id'])
    elif table_name == "patients":
        return str(row["subject_id"])
    elif table_name == "labevents":
        return str(row["labevent_id"])
    elif table_name == "procedures_icd":
        elem1 = str(row["hadm_id"])
        elem2 = str(row["seq_num"])
        elem3 = str(row["icd_code"].upper())
        return "-".join([elem1, elem2, elem3])
    elif table_name == "chartevents":
        elem1 = str(row["stay_id"])
        elem2 = str(row["charttime"])
        elem3 = str(row["itemid"])
        elem4 = str(row["value"])
        return "-".join([elem1, elem2, elem3, elem4])
    elif table_name == "inputevents":
        return f"{row['stay_id']}-{row['orderid']}-{row['itemid']}"
    elif (table_name == "microbiologyevents") and (resource_type == "ObservationMicroTest"):
        elem1 = str(row["micro_specimen_id"])
        elem2 = str(row["test_itemid"])
        return "-".join([elem1, elem2])
    elif (table_name == "microbiologyevents") and (resource_type == "ObservationMicroOrg"):
        elem1 = str(row["test_itemid"])
        elem2 = str(row["micro_specimen_id"])
        elem3 = str(int(row["org_itemid"])) if pd.notna(row["org_itemid"]) else "NULL"
        return "-".join([elem1, elem2, elem3])
    elif (table_name == "microbiologyevents") and (resource_type == "ObservationMicroSusc"):
        return str(row['microevent_id'])
    elif (table_name == "microbiologyevents") and (resource_type == "Specimen"):
        return str(row['micro_specimen_id'])
    elif table_name == "outputevents":
        elem1 = str(row["stay_id"])
        elem2 = str(row["charttime"])
        elem3 = str(row["itemid"])
        return "-".join([elem1, elem2, elem3])
    elif table_name == "transfers":
        return(str(row["careunit"]))
    
def convert_table_id_to_fhir_id(resource_type: str, table_id: str):
    print(resource_type, table_id)
    
    mimiciv_uuid5 = uuid.uuid5(uuid.NAMESPACE_OID, 'MIMIC-IV')
    resource_uuid5 = uuid.uuid5(mimiciv_uuid5, resource_type)
    if table_id is not None :
        uuid5 = str(uuid.uuid5(resource_uuid5, table_id))
    else:
        uuid5 = None
    return uuid5

def fix_datetime_3args(sql: str) -> str:
    """
    Rewrites all 3-argument datetime() calls like
    datetime('2021-12-31', 'start of month', '-1 month') or
    datetime('2021-12-31', 'start of year', '-1 year')
    into nested 2-argument calls:
    datetime(datetime('2021-12-31', 'start of month'), '-1 month') or
    datetime(datetime('2021-12-31', 'start of year'), '-1 year')
    so that sqlglot can parse it.
    """
    pattern = re.compile(
        r"datetime\(\s*([^,]+),\s*'(start of (month|year|day))'\s*,\s*'(-?\d+ \w+)'\s*\)", 
        re.IGNORECASE
    )
    return pattern.sub(r"datetime(datetime(\1, '\2'), '\4')", sql)


def extract_outer_subselects(sql):
    """
    Extracts only the top-level scalar subqueries from the SELECT clause of a SQL query.

    This function is designed to handle cases where the SELECT clause contains scalar subqueries 
    (e.g., SELECT (SELECT ...) - (SELECT ...)). It returns only the immediate subqueries at the 
    top level of the SELECT expression, excluding any subqueries that are nested deeper within 
    those subqueries (e.g., in WHERE or IN clauses).

    Args:
        sql (str): The full SQL query string.

    Returns:
        List[str]: A list of SQL subquery strings that are direct scalar subqueries of the 
        top-level SELECT clause.
    """
    formatted_query = sqlparse.format(sql, reindent=True, keyword_case='upper')
    fixed_query = fix_datetime_3args(formatted_query)
    parsed = sqlglot.parse_one(fixed_query)

    outer_subqueries = []

    # Go through each expression in the SELECT clause
    for expr in parsed.expressions:
        # If the expression is math (e.g., subquery - subquery), go deeper
        for child in expr.find_all(Subquery):
            # Only add subquery if its parent is the top-level expression
            if child.parent == expr:
                outer_subqueries.append(child.sql())
    outer_subqueries_transformed = []
    # transform outer subqueries from SELECT X FROM ... to SELECT * FROM ...
    for query in outer_subqueries:
        select_index = query.find("SELECT")
        from_index = query.find("FROM")
        table_prefix = query[from_index:].split(" ")[1]
        new_query = query[:select_index] + \
            f"SELECT {table_prefix}.* " + \
            query[from_index:]
        outer_subqueries_transformed.append(new_query)
    return outer_subqueries_transformed

def extract_first_subquery(query):
    formatted_query = sqlparse.format(query, reindent=True, keyword_case='upper')
    fixed_query = fix_datetime_3args(formatted_query)
    parsed = sqlglot.parse_one(fixed_query)

    # Look for the first subquery (a subquery is a SELECT within another SELECT, WHERE, etc.)
    for node in parsed.walk():
        if isinstance(node, sqlglot.expressions.Subquery):
            return node.this.sql(dialect="sqlite")  # or your preferred dialect
    return None  # No subquery found

# Replace SELECT statement with * to get the relevant IDs: 
def replace_sql_query(query):
    if query == "null":
        return "null"
    # Special case for d_icd_diagnoses table queries
    if re.sub(r'\n+', ' ', query).strip().startswith("SELECT d_icd_diagnoses.long_title FROM d_icd_diagnoses"):
        # getting only the center query which queries from diagnoses_icd table
        start_index = query.find("(") + 1
        end_index = query.rfind(")")
        query = query[start_index:end_index]
    
    # Special case for d_icd_procedures table queries
    if re.sub(r'\n+', ' ', query).strip().startswith("SELECT d_icd_procedures.long_title FROM d_icd_procedures"):
        # getting only the center query which queries from procedures_id table
        query = extract_first_subquery(query)

    # If the query has scalar subqueries in SELECT clause
    # e.g., SELECT (SELECT ...) - (SELECT ...)
    scalar_subqueries = extract_outer_subselects(query)

    if scalar_subqueries:
        #print("Scalar subqueries found in SELECT clause:")
        select_wrapped = [f"SELECT * FROM {sq.strip()} AS sub{i+1}" for i, sq in enumerate(scalar_subqueries)]
        return "\nUNION ALL\n".join(select_wrapped)

    # TODO: Could do something smarter with SQL parser?
    # parser = Parser(query)
    # # Assume main select that is returned is the first elem in the 'select' key list
    select_index = query.find("SELECT")
    from_index = query.find("FROM")
    # We cannot assume the table name is the first word after FROM
    # as there are many nested subqueries in this dataset

    table_prefix = query[from_index:].split(" ")[1]
    if table_prefix.startswith("("):
        # For now, don't use alias in this case
        # We might want to use it in the future
        new_query = query[:select_index] + \
            "SELECT * " + \
            query[from_index:]
    else: 
        new_query = query[:select_index] + \
            f"SELECT {table_prefix}.* " + \
            query[from_index:]
    return new_query

def build_med_id(row):
    parts = []
    if row.get('drug'):
        parts.append(row['drug'])
    if row.get('formulary_drug_cd') not in ['', None]:
        parts.append(row['formulary_drug_cd'])
    if row.get('ndc') not in ['0', '', None]:
        ndc_str = re.sub(r"\.[^.]*$", "", str(row["ndc"])) if row["ndc"] is not None else None
        # Need to add 0 padding to ndc_str if it's less than 11 digits
        elem = ndc_str.zfill(11) if ndc_str is not None and ndc_str != "0" and ndc_str != "" else ""
        if elem != '':
            parts.append(elem)
    return '--'.join([str(p) for p in parts])


def compute_med_ids(df):
    """Add med_id column to a prescription DataFrame."""
    df = df.copy()
    df['med_id'] = df.apply(build_med_id, axis=1)
    return df

def get_resource_namespace_uuid(resource_type):
    """Generate the resource-level UUID namespace (e.g., for MedicationPrescriptions)."""
    mimiciv_uuid = uuid.uuid5(uuid.NAMESPACE_OID, 'MIMIC-IV')
    return uuid.uuid5(mimiciv_uuid, resource_type)

def compute_medmix_mapping(df, resource_type='MedicationMix'):
    """Generate medmix_id and UUIDs for each pharmacy_id group using chained UUIDv5."""
    df = df.copy()
    resource_namespace = get_resource_namespace_uuid(resource_type)

    # Define drug_type sort order
    drug_type_order = {'MAIN': 3, 'BASE': 2, 'ADDITIVE': 1}
    df['drug_type_priority'] = df['drug_type'].map(drug_type_order).fillna(0)

    # Sort rows within each group
    df = df.sort_values(['pharmacy_id', 'drug_type_priority', 'drug'],
                        ascending=[True, False, True])


    # Generate medmix_id: ordered concatenation of med_id per group
    medmix_df = (
        df.groupby('pharmacy_id')['med_id']
        .apply(lambda med_ids: '_'.join(med_ids))
        .reset_index(name='medmix_id')
    )

    # UUIDv5 generation with chained namespace
    medmix_df['fhir_id'] = medmix_df['medmix_id'].apply(
        lambda mid: str(uuid.uuid5(resource_namespace, mid))
    )
    medmix_df['table_id'] = medmix_df['medmix_id']
    return medmix_df

def assign_medmix_ids(df_prescriptions, resource_type='MedicationMix'):
    """
    Main function to assign medmix_id and UUID per row using chained UUID logic.
    """
    df = compute_med_ids(df_prescriptions).drop(['fhir_id', 'table_id'],
                                                axis=1,
                                                errors='ignore')
    medmix_map = compute_medmix_mapping(df, resource_type)
    df = df.merge(medmix_map, on='pharmacy_id', how='left')
    return df.drop(columns=['drug_type_priority'], errors='ignore')