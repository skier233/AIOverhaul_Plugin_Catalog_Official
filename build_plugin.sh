#!/bin/bash
# Build script to copy skier_aitagging plugin from catalog to Stash-AIServer
# This script is gitignored and copies plugin files for testing

set -e

CATALOG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_NAME="skier_aitagging"
SOURCE_DIR="$CATALOG_DIR/$PLUGIN_NAME"
TARGET_DIR="$CATALOG_DIR/../Stash-AIServer/plugins/$PLUGIN_NAME"

echo "Building plugin: $PLUGIN_NAME"
echo "Source: $SOURCE_DIR"
echo "Target: $TARGET_DIR"

# Check source exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "ERROR: Source directory does not exist: $SOURCE_DIR" >&2
    exit 1
fi

# Create target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Copy all files (exclude .git, __pycache__, .pyc files)
echo ""
echo "Copying files..."

copied=0
skipped=0

find "$SOURCE_DIR" -type f \( -name "*.py" -o -name "*.yml" -o -name "*.csv" -o -name "*.txt" -o -name "*.md" \) ! -path "*/.git/*" ! -path "*/__pycache__/*" ! -name "*.pyc" | while read -r file; do
    relative_path="${file#$SOURCE_DIR/}"
    target_path="$TARGET_DIR/$relative_path"
    target_parent="$(dirname "$target_path")"
    
    # Create parent directory if needed
    mkdir -p "$target_parent"
    
    # Copy file
    if cp "$file" "$target_path" 2>/dev/null; then
        echo "  ✓ $relative_path"
        ((copied++))
    else
        echo "  ✗ $relative_path - Failed to copy"
        ((skipped++))
    fi
done

echo ""
echo "Build complete!"
echo "  Copied: $copied files"
if [ $skipped -gt 0 ]; then
    echo "  Skipped: $skipped files"
fi
echo ""
echo "Plugin is now available at: $TARGET_DIR"
