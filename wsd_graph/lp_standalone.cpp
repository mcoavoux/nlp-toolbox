#include "lp.h"
#include <fstream>
#include <getopt.h>


int main(int argc, char* argv[]){

  string path="json.txt";
  int k = -1;
  float e = 0.0;
  const struct option longopts[] =
  {
    {"outfile",     required_argument,  0, 'o'},
    {"knn",     required_argument,  0, 'k'},
    {"enn",     required_argument,  0, 'e'},
    {0,0,0,0}
  };
  int c = 0;
  while((c = getopt_long(argc, argv, "o:k:e:",longopts,NULL)) != -1){
    switch(c){
    case 'o':
      path = string(optarg);
      break;
    case 'k':
      k = stoi(optarg);
      break;
    case 'e':
      e = stof(optarg);
      break;
    default:
      break;
  }
 }

 if (optind < argc){//we have still at least one positional argument
   WSDdataset data(argv[optind]); // /Users/bcrabbe/Desktop/LP.in
   WSDGraph G;
   Proximity *prox = new Proximity();
   data.get_graph(G,prox);
   delete prox;
   if (e > 0){G.eps_nn(e);}
   if (k > 0){G.knn(k);}
   G.label_propagation(0.0000001);
   wofstream out(path);
   G.json(out);
   out.close();      
  }else{
   cerr << "missing input file. aborting."<< endl;
   cerr << "usage : propagate <input_filename>"<<endl;
   exit(1);
  }
}
