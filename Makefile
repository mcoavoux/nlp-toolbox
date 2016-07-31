

all: tok wsd

wsd:
	$(MAKE) -C wsd_graph

tok:
	$(MAKE) -C tokenizer

clean:
	$(MAKE) clean -C wsd_graph
	$(MAKE) clean -C tokenizer