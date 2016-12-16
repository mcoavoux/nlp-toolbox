#include "tokenizer.h"
#include <iostream>
#include <fstream>
#include <getopt.h>


void display_help_msg(){

  cerr << "usage : tokenizer <input_filename> <output_filename>"<< endl;
  cerr << "--help        displays this message and exits." << endl;
  cerr << "--strong-compounds     dictionary of compounds to be merged without separator." << endl;
  cerr << "--weak-compounds       dictionary of compounds to be merged with a separator '_'." << endl;
  cerr << "--prefixes             dictionary of prefixes to be merged without a separator to the following token." << endl;
  cerr << "--column               outputs tokens in columns" << endl;
  cerr << "--sentence             also performs sentence segmentation" << endl;
}


bool is_file_exist(const char *fileName)
{
    std::ifstream infile(fileName);
    return infile.good();
}


int main(int argc, char* argv[]){

  string strongcpd= "";
  string weakcpd= "";
  string prefixcpd= "";
  bool debug = false;
  bool column = false;
  bool sent_boundary = false;
  const struct option longopts[] =
  {
    {"debug",     no_argument,  0, 'd'},
    {"column",     no_argument,  0, 'c'},
    {"sentence",     no_argument,  0, 's'},
    {"strong-compounds",     no_argument,  0, 'S'},
    {"weak-compounds",     no_argument,  0, 'W'},
    {"prefixes",     no_argument,  0, 'P'},
    {0,0,0,0}
  };

  int c = 0;
  while((c = getopt_long(argc, argv, "csdS:W:P:",longopts,NULL)) != -1){
    switch(c){
    case 'S':
      strongcpd = optarg;
      break;
    case 'W':
      weakcpd = optarg;
      break;
    case 'P':
      prefixcpd = optarg;
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
   if(!strongcpd.empty()){tok.add_strong_cpd_dictionary(string("strong"),strongcpd.c_str());}
   if(!weakcpd.empty()){tok.add_weak_cpd_dictionary(string("cpd"),weakcpd.c_str());}
   if(!prefixcpd.empty()){tok.add_prefix_dictionary(string("prefixes-hyphen"),prefixcpd.c_str());}
   tok.compile();
   tok.batch_tokenize_file(argv[optind],argv[optind+1],column,sent_boundary);      
  }else{
   cerr << "missing input or output files. aborting."<< endl;
   display_help_msg();
   exit(1);
  }
}
