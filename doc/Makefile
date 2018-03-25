thesis.pdf: thesis.tex
	latexrun --latex-cmd xelatex --bibtex-cmd=biber --latex-args="-synctex=1" thesis.tex

thesis.tex: figures/graphics.sketch
	sketchtool export artboards figures/graphics.sketch --output=figures/sketch
	jupyter nbconvert --execute --to notebook --inplace ../notebooks/figures/styles.ipynb