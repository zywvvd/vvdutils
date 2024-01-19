#!/bin/bash
cd mtutils

git pull
git add -A
git commit -m update-utils
git push

cd ..

echo "@@ updating tiny version..."
python version_update.py

echo "@@ making sdist directory..."
python setup.py sdist

echo "@@ uploading dist..."
twine upload dist/*

echo "@@ removing directory..."
rm -rf dist
rm -rf mtutils.egg-info
