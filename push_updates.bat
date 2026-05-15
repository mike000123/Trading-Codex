@echo on

echo ADDING FILES
git add .

echo COMMITTING
git commit -m "full overwrite"

echo FORCE PUSHING (WARNING: overwrites remote)
git push --force

pause