
g++ -I/Users/bcrabbe/Desktop/WSD_GRAPH/ -I/opt/local/include -L/opt/local/lib -std=c++11 -O3 -o tokenizer tokenizer.cpp -lafcpu -lboost_regex-mt
cat prefixes-hyphen.dic | python uppercase_add.py > prefixes-hyphen-full.dic
cat prefixes-quotes.dic| python uppercase_add.py > prefixes-quotes-full.dic
cat abbreviations.dic| python uppercase_add.py > abbreviations-full.dic
cat weakcpd.dic| python uppercase_add.py > weakcpd-full.dic
#do nothing for strong-cpd.dic
#todo: emojis