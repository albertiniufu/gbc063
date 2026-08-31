# Compilação XeLaTeX: latexmk resolve as múltiplas passadas sozinho;
# -cd compila a partir do diretório do .tex (PDF e temporários ficam em aulaNN/)
LATEXMK = latexmk -cd -xelatex -interaction=nonstopmode

# Regra genérica: .tex -> .pdf (slides, handouts e documentos avulsos)
%.pdf: %.tex
	$(LATEXMK) $<

# Aulas descobertas a partir dos diretórios aulaNN
AULAS   := $(patsubst aula%,%,$(notdir $(wildcard aula*)))
TARGETS := $(addprefix aula,$(AULAS))

# make aulaNN recompila slides + handout da aula NN (make -B aulaNN força)
# Regras explícitas geradas (make 4.4 não aceita dois % num pré-requisito padrão)
define AULA_RULE
aula$(1): aula$(1)/aula$(1).pdf aula$(1)/handout-aula$(1).pdf
endef
$(foreach n,$(AULAS),$(eval $(call AULA_RULE,$(n))))

# Compila todas as aulas
all: $(TARGETS)

# Extensões de arquivos temporários gerados pela compilação LaTeX (beamer)
clean:
	find . -type f \
		\( -name '*.aux' -o -name '*.log' -o -name '*.out' \
		 -o -name '*.toc' -o -name '*.nav' -o -name '*.snm' \
		 -o -name '*.vrb' -o -name '*.bbl' -o -name '*.blg' \
		 -o -name '*.fls' -o -name '*.fdb_latexmk' \
		 -o -name '*.synctex.gz' -o -name '*.xdv' -o -name '*.dvi' \) \
		-delete

.PHONY: clean all $(TARGETS)
