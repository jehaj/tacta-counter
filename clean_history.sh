#!/bin/bash
set -e

# Use single quotes for the outer shell command and double quotes for the inner script
git filter-repo --force --file-filter '
    if [[ "$FILE" == *.ipynb ]]; then
        nbstripout "$FILE" > "$FILE.tmp" && mv "$FILE.tmp" "$FILE"
    fi
'

read -p 'Are you sure you want to force push? This will overwrite remote history. (y/N) ' -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo 'Aborted.'
    exit 1
fi

git push origin --force --all
