#include "tokenizer.h"
#include <iostream>
#include <getopt.h>


int main(int argc, char* argv[]) {

  string path=".";
  bool debug = false;
  bool column = false;
  const struct option longopts[] =
  {
    {"path",     required_argument,  0, 'p'},
    {"debug",     no_argument,  0, 'd'},
    {"column",     no_argument,  0, 'c'},
    {0,0,0,0}
  };
  int c = 0;
  while((c = getopt_long(argc, argv, "cdp:",longopts,NULL)) != -1){
    switch(c){
    case 'p':
      path = optarg;
      break;
    case 'd':
      debug = true;
      break;
    case 'c':
      column = true;
      break;
    default:
      break;
  }
 }

 if (optind+1 < argc){//we have still at least two positional arguments
   Tokenizer tok;
   tok.add_strong_cpd_dictionary(string("strong"),(path+string("/strong-cpd.dic")).c_str());
   tok.add_strong_cpd_dictionary(string("abbr"),(path+string("/abbreviations-full.dic")).c_str());
   tok.add_weak_cpd_dictionary(string("cpd"),(path+string("/weakcpd-full.dic")).c_str()); 
   tok.add_prefix_dictionary(string("prefixes-hyphen"),(path+string("/prefixes-hyphen-full.dic")).c_str());
   tok.add_prefix_dictionary(string("prefixes-quote"),(path+string("/prefixes-quotes-full.dic")).c_str());
   tok.compile();
   tok.batch_tokenize_file(argv[optind],argv[optind+1],column);      
  }else{
   cerr << "missing input or output files. aborting."<< endl;
   cerr << "usage : tokenizer <input_filename> <output_filename>"<<endl;
   exit(1);
  }
}
