#!/bin/bash
# Download MIMIC-IV-on-FHIR demo (100 real de-identified patients, open access) and load the
# 8 gold resource types into the running Medplum.
# Timing note: ~1h on a t3.xlarge (4 vCPU) — this figure is EC2-measured and NOT verified on a
# laptop Docker path. On a 2-core Docker Desktop VM, lower the concurrency: `W=4 bash load_mimic.sh`.
# Idempotent (PUT preserves ids). Tunables: W (parallel workers, default 16), BATCH (bundle size, 150).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
if ! command -v wget >/dev/null 2>&1; then
  echo "ERROR: 'wget' is required to download the MIMIC demo but was not found." >&2
  echo "       Install it (macOS: 'brew install wget'; Debian/Ubuntu: 'apt-get install wget')." >&2
  exit 1
fi
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
NF=$(echo $PATHS | wc -w | tr -d ' ')
if [ "$NF" -eq 0 ]; then
  echo "ERROR: found 0 MIMIC ndjson.gz files under $HOME/mimic/fhir/ — the download failed or the" >&2
  echo "       physionet layout changed. Expected files like \$HOME/mimic/fhir/MimicPatient.ndjson.gz." >&2
  echo "       Without this guard the loader would silently load nothing and the eval would score all-wrong." >&2
  exit 1
fi
echo "loading $NF files into Medplum..."
W="${W:-16}" BATCH="${BATCH:-150}" python3 "$HERE/bulk_load.py" $PATHS
echo "done. sanity check: 'curl -s \"\$MEDPLUM_BASE_URL/fhir/R4/Patient?_summary=count\" -H \"Authorization: Bearer \$(python3 $HERE/get_token.py)\"' should report ~100 patients."
