@echo off
chcp 65001 >nul 2>&1
title BlobVision - Tauri
cd /d "%~dp0tauri"

echo.
echo  BlobVision - Tauri - demarrage...
echo  Interface en mode dev - premiere generation = chargement modeles, 1 a 3 min.
echo.

set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"

where cargo >nul 2>&1
if errorlevel 1 (
    echo Rust/Cargo introuvable. Installe-le via https://rustup.rs puis relance.
    pause
    exit /b 1
)

if not exist node_modules (
    echo  Installation des dependances frontend, premiere fois seulement...
    call npm install
    if errorlevel 1 (
        echo.
        echo Echec de "npm install" - voir l'erreur ci-dessus.
        pause
        exit /b 1
    )
)

call npm run tauri dev
if errorlevel 1 (
    echo.
    echo BlobVision - Tauri a plante - voir l'erreur ci-dessus.
)
echo.
echo  BlobVision - Tauri ferme. Appuie sur une touche pour fermer cette fenetre.
pause >nul
