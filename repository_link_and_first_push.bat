@echo off
cd /d %~dp0

echo ==============================
echo GIT AUTO SETUP + PUSH START
echo ==============================

:: 1. Init git if not already
if not exist .git (
    echo Initializing git repo...
    git init
)

:: 2. Ensure branch is main
git branch -M main

:: 3. Remove old origin if exists
git remote remove origin 2>nul

:: 4. Add correct GitHub repo
git remote add origin https://github.com/mike000123/Trading-Codex.git

:: 5. Add all files
echo Adding files...
git add .

:: 6. Commit (safe if nothing changed)
echo Committing...
git commit -m "Auto setup commit %date% %time%" 2>nul

:: 7. Force push (avoids pull/merge issues)
echo Pushing to GitHub...
git push -u origin main --force

echo ==============================
echo DONE - REPO LINKED & PUSHED
echo ==============================
pause