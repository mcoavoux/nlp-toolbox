#include "tokenizer.h"
#include <iostream>
#include <getopt.h>

void display_help_msg(){
  cerr << "usage : tokenizer <input_filename> <output_filename>"<< endl;
  cerr << "--help        displays this message and exits." << endl;
  cerr << "--path        path to param files directory (default = '.')" << endl;
  cerr << "--column      outputs tokens in columns" << endl;
  cerr << "--sentence    performs sentence segmentation" << endl;
}


int main(int argc, char* argv[]){

  string path=".";
  bool debug = false;
  bool column = false;
  bool sent_boundary = false;
  const struct option longopts[] =
  {
    {"path",     required_argument,  0, 'p'},
    {"debug",     no_argument,  0, 'd'},
    {"column",     no_argument,  0, 'c'},
    {"sentence",     no_argument,  0, 's'},
    {0,0,0,0}
  };
  int c = 0;
  while((c = getopt_long(argc, argv, "csdp:",longopts,NULL)) != -1){
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
    case 's':
      sent_boundary = true;
      break;
    case 'h':
      display_help_msg();
      exit(0); 
    default:
      break;
  }
 }

 if (optind+1 < argc){//we have still at least two positional arguments
   Tokenizer tok;
   tok.add_strong_cpd_dictionary(string("strong"),(path+string("/strong-cpd-full.dic")).c_str());
   tok.add_weak_cpd_dictionary(string("cpd"),(path+string("/weakcpd-full.dic")).c_str()); 
   tok.add_prefix_dictionary(string("prefixes-hyphen"),(path+string("/prefixes-full.dic")).c_str());
   tok.compile();
   tok.batch_tokenize_file(argv[optind],argv[optind+1],column,sent_boundary);      
  }else{
   cerr << "missing input or output files. aborting."<< endl;
   display_help_msg();
   exit(1);
  }
}
