TARGET   = paper
PDFLATEX = pdflatex
DOCS     = paper
BIBINPUTS = sc20_paper.bib

paper:
	${PDFLATEX} ${TARGET}
	bibtex ${TARGET}
	${PDFLATEX} ${TARGET}
	${PDFLATEX} ${TARGET}
	make clean

clean:
	rm -f ${TARGET}.out ${TARGET}.aux ${TARGET}.bbl ${TARGET}.blg ${TARGET}.log
	rm -f ${TARGET}.lof ${TARGET}.fff *.gz *.fls *.fdb_latexmk
	rm -rf *_latexmk_temp
	rm -f *criticMarkupProcessed* tti_wri_convertScholmd.bash script_name

realclean:
	rm -f *.pdf

default: paper

.PHONY: simple
