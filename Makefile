

all: nnlib ppa_exp tok wsd

wsd:
	$(MAKE) -C wsd_graph

tok:
	$(MAKE) -C tokenizer

nnlib: 
	$(MAKE) -C nn

ppa_exp : 
	$(MAKE) -C ppa

clean:
	$(MAKE) clean -C wsd_graph
	$(MAKE) clean -C tokenizer
	$(MAKE) clean -C nn
	$(MAKE) clean -C ppa
