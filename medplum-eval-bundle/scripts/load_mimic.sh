#!/bin/bash
# Download MIMIC-IV-on-FHIR demo (100 real de-identified patients, open access) and load the
# 8 gold resource types into the running Medplum. ~1h on t3.xlarge. Idempotent (PUT preserves ids).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$HOME/mimic" && cd "$HOME/mimic"
if [ ! -d fhir ]; then
  echo "downloading MIMIC-IV-on-FHIR demo..."
  wget -q -r -np -nH --cut-dirs=3 -R "index.html*,robots.txt" -A "ndjson,gz,json" \
    https://physionet.org/files/mimic-iv-fhir-demo/2.1.0/
fi
# 8 gold types only (drop MedicationAdministration/Dispense/Specimen/MedicationStatement); chartevents last
FILES="MimicOrganization MimicLocation MimicPatient \
MimicEncounter MimicEncounterED MimicEncounterICU \
MimicMedication MimicMedicationMix MimicCondition MimicConditionED \
MimicProcedure MimicProcedureICU MimicProcedureED MimicMedicationRequest \
MimicObservationLabevents MimicObservationDatetimeevents MimicObservationOutputevents \
MimicObservationMicroTest MimicObservationMicroOrg MimicObservationMicroSusc \
MimicObservationVitalSignsED MimicObservationED MimicObservationChartevents"
PATHS=""
for f in $FILES; do [ -f "fhir/$f.ndjson.gz" ] && PATHS="$PATHS $HOME/mimic/fhir/$f.ndjson.gz"; done
echo "loading $(echo $PATHS | wc -w) files into Medplum..."
W="${W:-16}" BATCH="${BATCH:-150}" python3 "$HERE/bulk_load.py" $PATHS
