import os

# Configure LiteLLM to reduce noise
try:
    import litellm
    litellm.suppress_debug_info = True
    litellm.set_verbose = False
except ImportError:
    pass

# Directory paths
MIMIC_IV_BASE_DIR = "ehrsql-2024/data/mimic_iv"
MIMIC_IV_TRAIN_DIR = os.path.join(MIMIC_IV_BASE_DIR, "train")
MIMIC_IV_VALID_DIR = os.path.join(MIMIC_IV_BASE_DIR, "valid")
MIMIC_IV_TEST_DIR = os.path.join(MIMIC_IV_BASE_DIR, "test")

DATA_PATHS = [MIMIC_IV_TRAIN_DIR, MIMIC_IV_VALID_DIR, MIMIC_IV_TEST_DIR]

# Database paths
UNPROCESSED_DB_PATH = 'data/mimic_iv_unprocessed_raw.sqlite'
PROCESSED_DB_PATH = 'data/mimic_iv_unprocessed.sqlite'

# Output files
QUESTION_ANSWER_OUTPUT_CSV = 'final_dataset/questions_answers_sql.csv'
QUESTION_ANSWER_FHIR_OUTPUT_CSV = 'final_dataset/questions_answers_fhir.csv'
QUESTION_SQL_FHIR_OUTPUT_CSV = "final_dataset/questions_answers_sql_fhir.csv"
VALUE_MAPPING_FILE = 'value_mapping_valid_natural.json'

# Vital signs templates
VITAL_TEMPLATES = [
    "What is the change in the {vital_name} of patient {patient_id} from the [time_filter_exact2] value measured [time_filter_global2] compared to the [time_filter_exact1] value measured [time_filter_global1]?",
    "Is the {vital_name} of patient {patient_id} [time_filter_exact2] measured [time_filter_global2] [comparison] than the [time_filter_exact1] value measured [time_filter_global1]?",
    "What was the [time_filter_exact1] measured {vital_name} of patient {patient_id} [time_filter_global1]?",
    "What was the [agg_function] {vital_name} of patient {patient_id} [time_filter_global1]?",
    "When was the [time_filter_exact1] time that patient {patient_id} had a {vital_name} measured [time_filter_global1]?",
    "When was the [time_filter_exact1] time that the {vital_name} of patient {patient_id} was [comparison] than {vital_value} [time_filter_global1]?",
    "When was the [time_filter_exact1] time that patient {patient_id} had the [sort] {vital_name} [time_filter_global1]?",
    "Has_verb the {vital_name} of patient {patient_id} been ever [comparison] than {vital_value} [time_filter_global1]?",
    "Has_verb the {vital_name} of patient {patient_id} been normal [time_filter_global1]?",
    "List the [unit_average] [agg_function] {vital_name} of patient {patient_id} [time_filter_global1]."
]

# Weight templates
WEIGHT_TEMPLATES = [
    "List the [unit_average] [agg_function] weight of patient {patient_id} [time_filter_global1].",
    "What was the [time_filter_exact1] measured weight of patient {patient_id} [time_filter_global1]?",
    "What is the change in the weight of patient {patient_id} from the [time_filter_exact2] value measured [time_filter_global2] compared to the [time_filter_exact1] value measured [time_filter_global1]?"
]

# Height templates
HEIGHT_TEMPLATES = [
    'What was the [time_filter_exact1] measured height of patient {patient_id} [time_filter_global1]?'
]

# Gender templates
GENDER_TEMPLATES = [
    'What are_verb the top [n_rank] frequently prescribed drugs that {gender} patients aged [age_group] were prescribed [time_filter_within] after having been diagnosed with {diagnosis_name} [time_filter_global1]?'
]

# Emergency room templates
ER_TEMPLATES = [
    "Has_verb patient {patient_id} been to an emergency room [time_filter_global1]?"
]

# Sum templates
SUM_TEMPLATES = [
    'What was the total amount of dose of {drug_name} that patient {patient_id} were prescribed [time_filter_global1]?',
    'What was the total volume of intake that patient {patient_id} received [time_filter_global1]?',
    'What was the total volume of output that patient {patient_id} had [time_filter_global1]?',
    'What was the total volume of {input_name} intake that patient {patient_id} received [time_filter_global1]?',
    'What was the total volume of {output_name} output that patient {patient_id} had [time_filter_global1]?'
]

# dose_val_rx
DOSE_VAL_RX_TEMPLATES = [
    'What was the dose of {drug_name} that patient {patient_id} was [time_filter_exact1] prescribed [time_filter_global1]?',
    'What was the total amount of dose of {drug_name} that patient {patient_id} were prescribed [time_filter_global1]?'
]

# Templates to remove
# Reason 1: totalamount not mapped to FHIR
# Reason 2: inputevents.starttime not mapped to FHIR
# Reason 3: # complex SQL structures that require separate, specialized handling (e.g., EXCEPT clauses, specific joins, volume calculations).
REMOVE_TEMPLATES = [
    'What was the total volume of {input_name} intake that patient {patient_id} received [time_filter_global1]?',
    'What was the total volume of intake that patient {patient_id} received [time_filter_global1]?',
    'List the [unit_average] [agg_function] volume of {input_name} intake that patient {patient_id} received [time_filter_global1].',
    'What is the difference between the total volume of intake and output of patient {patient_id} [time_filter_global1]?',
    'When was the [time_filter_exact1] intake time of patient {patient_id} [time_filter_global1]?',
    'Has_verb patient {patient_id} had any {input_name} intake [time_filter_global1]?',
    'How many [unit_count] have passed since the [time_filter_exact1] time patient {patient_id} had a {input_name} intake on the current ICU visit?',
    'What was the name of the intake that patient {patient_id} [time_filter_exact1] had [time_filter_global1]?',
    'When was the [time_filter_exact1] time that patient {patient_id} had a {input_name} intake [time_filter_global1]?',
    'Count the number of times that patient {patient_id} had a {input_name} intake [time_filter_global1].',
    'What is the new prescription of patient {patient_id} [time_filter_global2] compared to the prescription [time_filter_global1]?',
    'What was the name of the drug that patient {patient_id} was prescribed [time_filter_within] after having received a {procedure_name} procedure [time_filter_global1]?',
    'What was the name of the drug that patient {patient_id} was prescribed [time_filter_within] after having been diagnosed with {diagnosis_name} [time_filter_global1]?',
]

# Vital signs mapping
VITAL_SIGNS_MAPPING = {
    "Temperature Celsius": {
        "sql_value": "Temperature Celsius",
        "question_value": "body temperature"
    },
    "O2 saturation pulseoxymetry": {
        "sql_value": "O2 saturation pulseoxymetry", 
        "question_value": "SpO2"
    },
    "Heart Rate": {
        "sql_value": "Heart Rate",
        "question_value": None
    },
    "Respiratory Rate": {
        "sql_value": "Respiratory Rate",
        "question_value": None
    },
    "Arterial Blood Pressure systolic": {
        "sql_value": "Arterial Blood Pressure systolic",
        "question_value": "systolic blood pressure"
    },
    "Arterial Blood Pressure diastolic": {
        "sql_value": "Arterial Blood Pressure diastolic",
        "question_value": "diastolic blood pressure"
    },
    "Arterial Blood Pressure mean": {
        "sql_value": "Arterial Blood Pressure mean",
        "question_value": "mean blood pressure"
    }
}

# Text columns for value mapping
TEXT_COLUMNS = [
    'drug_route', 'lab_name', 'procedure_name', 'careunit', 'diagnosis_name',
    'spec_name', 'input_name', 'output_name', 'admission_route', 'drug_name',
    'drug_name1', 'drug_name2', 'drug_name3'
]

# Frequent term mapper
FREQUENT_TERM_MAPPER = {
    "'Temperature Celsius'": "Search for 'Temperature Celsius' to find body temperature.",
    "'O2 saturation pulseoxymetry'": "Search for 'O2 saturation pulseoxymetry' to find SpO2.",
    "'Arterial Blood Pressure systolic'": "Search for 'Arterial Blood Pressure systolic' to find systolic blood pressure.",
    "'Arterial Blood Pressure diastolic'": "Search for 'Arterial Blood Pressure diastolic' to find diastolic blood pressure.",
    "'Arterial Blood Pressure mean'": "Search for 'Arterial Blood Pressure mean' to find mean blood pressure.",
    "'Daily Weight'": "Search for 'Daily Weight' to find weight.",
    "'Height (cm)'": "Search for 'Height (cm)' to find height.",
    "'M'": "Search for 'M' to find male.",
    "'F'": "Search for 'F' to find female."
}

# Label mapper for drug search locations
LABEL_MAPPER = {
    'micafungin': 'inputevents', 'propofol': 'inputevents', 'progesterone': 'prescriptions',
    'lr': 'inputevents', 'ambisome': 'inputevents', 'ceftaroline': 'inputevents',
    'epinephrine': 'inputevents', 'magnesium sulfate': 'inputevents', 'albumin 25%': 'inputevents',
    'caspofungin': 'inputevents', 'ciprofloxacin': 'inputevents', 'atropine': 'inputevents',
    'fentanyl': 'inputevents', 'dextrose 50%': 'inputevents', 'atovaquone': 'inputevents',
    'penicillin g potassium': 'inputevents', 'heparin': 'prescriptions', 'diltiazem': 'inputevents',
    'dilantin': 'inputevents', 'esmolol': 'inputevents', 'testosterone': 'labevents',
    'dopamine': 'inputevents', 'potassium phosphate': 'inputevents', 'albumin': 'labevents',
    'ensure': 'inputevents', 'verapamil': 'inputevents', 'procainamide': 'prescriptions',
    'ceftriaxone': 'inputevents', 'ceftazidime': 'inputevents', 'd5ns': 'inputevents',
    'citrate': 'inputevents', 'tobramycin': 'labevents', 'solution': 'inputevents',
    'phenylephrine': 'inputevents', 'primidone': 'prescriptions', 'vasopressin': 'inputevents',
    'doxycycline': 'inputevents', 'albumin 5%': 'inputevents', 'ribavirin': 'inputevents',
    'metronidazole': 'inputevents', 'multivitamins': 'inputevents', 'potassium': 'prescriptions',
    'tamiflu': 'inputevents', 'factor xiii': 'labevents', 'meropenem': 'inputevents',
    'oxycodone': 'labevents', 'protamine sulfate': 'inputevents', 'd5 1/2ns': 'inputevents',
    'rocuronium': 'inputevents', 'calcium chloride': 'inputevents', 'test': 'prescriptions',
    'phenobarbital': 'prescriptions', 'calcium gluconate': 'inputevents', 'linezolid': 'inputevents',
    'potassium chloride': 'inputevents', 'clindamycin': 'inputevents', 'adenosine': 'inputevents',
    'morphine sulfate': 'inputevents', 'chloroquine': 'inputevents', 'folic acid': 'inputevents',
    'amikacin': 'prescriptions', 'nafcillin': 'inputevents', 'daptomycin': 'inputevents',
    'd5lr': 'inputevents', 'sodium': 'labevents', 'quinidine': 'prescriptions',
    'moxifloxacin': 'inputevents', 'mannitol': 'inputevents', 'ampicillin': 'inputevents',
    'isoniazid': 'inputevents', 'fluconazole': 'inputevents', 'lithium': 'labevents',
    'keflex': 'inputevents', 'folate': 'inputevents', 'colistin': 'inputevents',
    'ranitidine': 'inputevents', 'argatroban': 'inputevents', 'dobutamine': 'inputevents',
    'cisatracurium': 'inputevents', 'dextrose 5%': 'inputevents', 'levofloxacin': 'inputevents',
    'factor viii': 'inputevents', 'acetaminophen': 'prescriptions', 'd': 'prescriptions',
    'lidocaine': 'prescriptions', 'theophylline': 'labevents', 'vancomycin': 'prescriptions',
    'methotrexate': 'prescriptions', 'aminophylline': 'inputevents', 'hydrochloric acid': 'inputevents',
    'nesiritide': 'inputevents', 'hydromorphone (dilaudid)': 'inputevents', 'octreotide': 'inputevents',
    'gentamicin': 'prescriptions', 'nicardipine': 'inputevents', 'rifampin': 'inputevents',
    'amiodarone': 'inputevents', 'tigecycline': 'inputevents', 'magnesium': 'labevents',
    'insulin': 'prescriptions', 'hydralazine': 'inputevents', 'labetalol': 'inputevents',
    'cyclosporine': 'inputevents', 'sodium bicarbonate 8.4%': 'inputevents', 'potassium acetate': 'inputevents',
    'aztreonam': 'inputevents', 'cefepime': 'inputevents', 'carbamazepine': 'labevents',
    'norepinephrine': 'inputevents', 'digoxin': 'prescriptions', 'milrinone': 'inputevents',
    'heparin sodium': 'inputevents', 'valproic acid': 'labevents', 'foscarnet': 'inputevents',
    'mannitol 20%': 'inputevents', 'd5 1/4ns': 'inputevents', 'phenytoin': 'labevents',
    'iron': 'labevents', 'acetylcysteine': 'inputevents', 'epoprostenol (veletri)': 'inputevents',
    'pyrazinamide': 'inputevents', 'sodium acetate': 'inputevents', 'acyclovir': 'inputevents',
    'oxacillin': 'inputevents', 'ketamine': 'inputevents', 'fosphenytoin': 'inputevents',
    'sterile water': 'inputevents', 'metoprolol': 'inputevents', 'nitroglycerin': 'inputevents',
    'erythromycin': 'inputevents', 'estradiol': 'labevents', 'voriconazole': 'inputevents',
    'pamidronate': 'inputevents', 'cefazolin': 'inputevents', 'azithromycin': 'inputevents',
    'fondaparinux': 'inputevents', 'lepirudin': 'inputevents', 'thiamine': 'inputevents',
    'thrombin': 'labevents', 'zinc': 'prescriptions'
}

# Value replacement patterns for question processing
VALUE_REPLACEMENT_PATTERNS = [
    (' {pre_value}?', ' {post_value}?'),
    (' {pre_value},', ' {post_value},'),
    (' {pre_value}-', ' {post_value}-'),
    (' {pre_value}.', ' {post_value}.'),
    ('"{pre_value}"', '"{post_value}"'),
    (" {pre_value}'s", " {post_value}'s"),
    (' {pre_value} ', ' {post_value} '),
    ('{pre_value_capitalized}', '{post_value_capitalized}')
]

QUESTION_FIXES_FILE = 'question_fixes_complete.json'

# Load question fixes from JSON file
def load_question_fixes():
    """Load question fixes from JSON file."""
    try:
        import json
        with open(QUESTION_FIXES_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

QUESTION_FIXES = load_question_fixes()