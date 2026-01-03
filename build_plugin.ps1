<#
.SYNOPSIS
    Build script to copy skier_aitagging plugin from catalog to Stash-AIServer for testing.

.DESCRIPTION
    This script copies the skier_aitagging plugin from the AIOverhaul_Plugin_Catalog_Official
    directory to the Stash-AIServer plugins directory. It also copies the frontend JavaScript
    file to the frontend dist directory for serving via the /plugins/ route.

.USAGE
    Run this script from the AIOverhaul_Plugin_Catalog_Official directory:
    
        .\build_plugin.ps1
    
    Or from any location:
    
        .\AIOverhaul_Plugin_Catalog_Official\build_plugin.ps1

.PREREQUISITES
    - The script must be located in: AIOverhaul_Plugin_Catalog_Official\
    - Source plugin directory must exist: AIOverhaul_Plugin_Catalog_Official\skier_aitagging\
    - Stash-AIServer must exist as a sibling directory: ..\Stash-AIServer\
    - Write permissions to Stash-AIServer\plugins\ and frontend\dist\plugins\

.WHAT IT DOES
    1. Copies all plugin files from catalog to Stash-AIServer\plugins\skier_aitagging\
    2. Excludes .git, __pycache__, .pyc files, and .gitignore
    3. Copies tag_list_editor.js to frontend\dist\plugins\skier_aitagging\ for web serving
    4. Creates target directories if they don't exist
    5. Overwrites existing files (uses -Force flag)

.NOTES
    This script is intended for local development/testing only.
    The plugin name is hardcoded as "skier_aitagging" in the script.
#>

$ErrorActionPreference = "Stop"

$catalogDir = $PSScriptRoot
$pluginName = "skier_aitagging"
$sourceDir = Join-Path $catalogDir $pluginName
$targetDir = Join-Path (Join-Path (Join-Path (Join-Path $catalogDir "..") "Stash-AIServer") "plugins") $pluginName

Write-Host "Building plugin: $pluginName" -ForegroundColor Cyan
Write-Host "Source: $sourceDir" -ForegroundColor Gray
Write-Host "Target: $targetDir" -ForegroundColor Gray

# Check source exists
if (-not (Test-Path $sourceDir)) {
    Write-Host "ERROR: Source directory does not exist: $sourceDir" -ForegroundColor Red
    exit 1
}

# Create target directory if it doesn't exist
if (-not (Test-Path $targetDir)) {
    Write-Host "Creating target directory: $targetDir" -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
}

# Get all files to copy (exclude .git, __pycache__, .pyc files)
$filesToCopy = Get-ChildItem -Path $sourceDir -File -Recurse | Where-Object {
    $_.FullName -notmatch '\.git' -and
    $_.FullName -notmatch '__pycache__' -and
    $_.Extension -ne '.pyc' -and
    $_.Name -ne '.gitignore'
}

Write-Host ""
Write-Host "Copying $($filesToCopy.Count) files..." -ForegroundColor Cyan

$copied = 0
$skipped = 0

foreach ($file in $filesToCopy) {
    $relativePath = $file.FullName.Substring($sourceDir.Length + 1)
    $targetPath = Join-Path $targetDir $relativePath
    $targetParent = Split-Path $targetPath -Parent
    
    # Create parent directory if needed
    if (-not (Test-Path $targetParent)) {
        New-Item -ItemType Directory -Path $targetParent -Force | Out-Null
    }
    
    # Copy file
    try {
        Copy-Item -Path $file.FullName -Destination $targetPath -Force
        Write-Host "  [OK] $relativePath" -ForegroundColor Green
        $copied++
    } catch {
        Write-Host "  [FAIL] $relativePath - $($_.Exception.Message)" -ForegroundColor Red
        $skipped++
    }
}

# Copy frontend JavaScript file to dist directory (for serving via /plugins/ route)
$frontendSource = Join-Path $sourceDir "tag_list_editor.js"
$stashAIServerDir = Join-Path (Join-Path $catalogDir "..") "Stash-AIServer"
$frontendDistDir = Join-Path (Join-Path (Join-Path $stashAIServerDir "frontend") "dist") "plugins"
$frontendDistPluginDir = Join-Path $frontendDistDir $pluginName
$frontendTarget = Join-Path $frontendDistPluginDir "tag_list_editor.js"

if (Test-Path $frontendSource) {
    Write-Host ""
    Write-Host "Copying frontend JavaScript file..." -ForegroundColor Cyan
    
    # Create frontend dist directory if it doesn't exist
    if (-not (Test-Path $frontendDistPluginDir)) {
        Write-Host "Creating frontend dist directory: $frontendDistPluginDir" -ForegroundColor Yellow
        New-Item -ItemType Directory -Path $frontendDistPluginDir -Force | Out-Null
    }
    
    try {
        # Copy to dist/plugins/pluginName/ (for /plugins/ route)
        Copy-Item -Path $frontendSource -Destination $frontendTarget -Force
        Write-Host "  [OK] tag_list_editor.js -> frontend/dist/plugins/$pluginName/" -ForegroundColor Green
        $copied++
        
        # Also ensure it's in the plugin directory itself (it should already be there from the main copy, but just in case)
        $pluginTarget = Join-Path $targetDir "tag_list_editor.js"
        if (-not (Test-Path $pluginTarget)) {
            Copy-Item -Path $frontendSource -Destination $pluginTarget -Force
            Write-Host "  [OK] tag_list_editor.js -> plugins/$pluginName/ (backup)" -ForegroundColor Green
        }
    } catch {
        Write-Host "  [FAIL] tag_list_editor.js - $($_.Exception.Message)" -ForegroundColor Red
        $skipped++
    }
} else {
    Write-Host ""
    Write-Host "  [SKIP] tag_list_editor.js not found in source" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Build complete!" -ForegroundColor Cyan
Write-Host "  Copied: $copied files" -ForegroundColor Green
if ($skipped -gt 0) {
    Write-Host "  Skipped: $skipped files" -ForegroundColor Yellow
}
Write-Host ""
Write-Host "Plugin is now available at: $targetDir" -ForegroundColor Cyan
Write-Host "Frontend file is now available at: $frontendTarget" -ForegroundColor Cyan

exit 0