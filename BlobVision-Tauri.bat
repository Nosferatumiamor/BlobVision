@echo off
chcp 65001 >nul 2>&1
title BlobVision
cd /d "%~dp0"

set "EXE=tauri\src-tauri\target\release\blobvision.exe"

if not exist "%EXE%" (
    echo.
    echo  BlobVision n'a pas encore ete compile en mode release.
    echo  Lance ceci une fois pour le construire ^(quelques minutes^) :
    echo.
    echo      cd tauri
    echo      npm run tauri build
    echo.
    echo  Ou utilise BlobVision-Tauri-Dev.bat en attendant ^(plus lent a chaque lancement^).
    echo.
    pause
    exit /b 1
)

echo.
echo  BlobVision - demarrage...
echo  Premiere generation = chargement modeles, 1 a 3 min. Les suivantes sont rapides.
echo.

"%EXE%"
if errorlevel 1 (
    echo.
    echo  BlobVision a plante ou n'a pas pu demarrer - voir l'erreur ci-dessus.
)
echo.
echo  BlobVision ferme. Appuie sur une touche pour fermer cette fenetre.
pause >nul
