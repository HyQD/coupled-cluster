#!/bin/bash

user_name="$GH_USER_NAME"
user_email="$GH_USER_EMAIL"

if [ -n "$(git status --porcelain)" ]; then
    git add .
    git -c user.name="$user_name" -c user.email="$user_email" commit -m "Black files"
    git push git@github.com:$TRAVIS_REPO_SLUG HEAD:$TRAVIS_BRANCH --quiet
fi
