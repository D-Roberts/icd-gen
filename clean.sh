rm -rf runs
rm -rf models
rm -rf output
rm -rf input

find . -type d -name "__pycache__" -exec rm -rf {} +

rm -rf src/enerdit/runs
rm -rf src/enerdit/models
rm -rf src/enerdit/output
rm -rf src/enerdit/input

rm -rf src/runs
rm -rf src/models
rm -rf src/output
rm -rf src/input
