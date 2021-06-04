#!/usr/bin/bash

# Generate README.md
pandoc Features.ipynb -s --extract-media=/tmp/pandoc --to gfm -o README.md

# Generate styled html
jupyter nbconvert Features.ipynb --to html --template pale-sand-navy 2> /dev/null
mv Features.html Comma.ai\ Speed\ Challenge.html

# Format the html
python3 -c "import os;exclude = ['    text-align: center;\n','<footer><a href=\"https://github.com/jelleschutter/nbconvert-theme-pale-sand-navy\">Pale Sand Navy Theme</a> by <a href=\"https://github.com/jelleschutter\">Jelle Schutter</a></footer>\n'];x = 'Comma.ai Speed Challenge.html';y = 'tmp.html';f = open(x);w = open(y, 'w+');[w.write(line) for line in f if not line in exclude];f.close();w.close();os.rename(y, x)"
