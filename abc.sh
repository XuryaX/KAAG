#!/bin/bash

# Prompt for repository name
read -p "Enter the repository name: " REPO_NAME

# Prompt for repository description (optional)
read -p "Enter the repository description (optional): " REPO_DESC

# Prompt for visibility (private/public)
read -p "Should the repository be private? (yes/no): " REPO_PRIVATE
if [ "$REPO_PRIVATE" = "yes" ]; then
    VISIBILITY="--private"
else
    VISIBILITY="--public"
fi

# Initialize a git repository
git init

# Add all files to the repo
git add .

# Commit the changes
git commit -m "Initial commit"

# Create a GitHub repository using gh CLI
gh repo create "$REPO_NAME" --description "$REPO_DESC" $VISIBILITY --source=. --remote=origin

# Push the current directory to the new GitHub repository
git push -u origin main

