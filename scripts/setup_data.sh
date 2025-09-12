# Setup script for EHRSQL-FHIR project
# Downloads MIMIC-IV demo data and EHRSQL dataset

echo "🔽 Setting up EHRSQL-FHIR Data"
echo "============================="

# Download the MIMIC-IV demo database
echo "📥 Downloading MIMIC-IV demo database..."
wget https://physionet.org/static/published-projects/mimic-iv-demo/mimic-iv-clinical-database-demo-2.2.zip

echo "📦 Extracting database..."
unzip mimic-iv-clinical-database-demo-2.2.zip

# Decompress all .gz files in their original folders
echo "🗜️  Decompressing CSV files..."
gunzip mimic-iv-clinical-database-demo-2.2/hosp/*.csv.gz
gunzip mimic-iv-clinical-database-demo-2.2/icu/*.csv.gz

# Create directory and move all CSV files
echo "📁 Organizing CSV files..."
mkdir -p ./data
find mimic-iv-clinical-database-demo-2.2/hosp -name "*.csv" -type f -exec mv {} ./data/ \;
find mimic-iv-clinical-database-demo-2.2/icu -name "*.csv" -type f -exec mv {} ./data/ \;
rm -rf mimic-iv-clinical-database-demo-2.2

# Clone ehrsql-2024 repository
echo "📥 Cloning EHRSQL-2024 dataset..."
git clone https://github.com/glee4810/ehrsql-2024.git