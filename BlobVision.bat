@echo off
chcp 65001 >nul 2>&1
title BlobVision
cd /d "%~dp0"
echo.
echo  BlobVision - demarrage...
echo  Fenetre native + chargement des modeles (1-3 min selon GPU).
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0app\run.ps1"
if errorlevel 1 (
    echo.
    echo BlobVision a plante - voir l'erreur ci-dessus.
    pause
)
