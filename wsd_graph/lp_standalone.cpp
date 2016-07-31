#include "lp.h"
#include <fstream>
#include <getopt.h>


int main(int argc, char *argv[]){

    int c;
    int knn = -1;
    float enn = -1;
    float reg_weights = 1.0;
    float ebd_weights = 1.0;
    bool nullifydiag = false;
    bool embeddings = false;
    const char *infile = "LP.in";
    const char *ebdfile;
    string outdir("analysis");

    const struct option longopts[] =
    {
      {"outfile",     required_argument,  0, 'o'},
      {"knn",     required_argument,  0, 'k'},
      {"enn",     required_argument,  0, 'e'},
      {"embeddings", required_argument, 0, 'E'},
      {"weight", required_argument, 0, 'w'},
      {"embeddings-weight", required_argument, 0, 'W'},
      {"null-diag", no_argument, 0, 'r'},
      {0,0,0,0}
    };
    while((c =  getopt(argc, argv, "f:k:e:w:W:E:r")) != EOF)
    {
        switch (c)
        {
	     case 'f':
	       infile = optarg;
	       break;
             case 'k':
	       knn = stoi(string(optarg));
               break;
             case 'e':
	       enn = stof(string(optarg));
	       break;
	     case 'E':
	       ebdfile = optarg;
	       embeddings = true;
	       break;
	     case 'w':
	       reg_weights = stof(string(optarg));
	       break;
	     case 'W':
	       ebd_weights = stof(string(optarg));
	       break;
	     case 'r':
	       nullifydiag = true;
	       break;
	     default:
	       cerr << "unknown option. aborting." << endl;
	       exit(1);
        }
    }
    if (optind < argc){
      outdir = argv[optind];
    }

    WSDdataset data;//(infile);
    if(embeddings){data.load_embeddings(ebdfile,reg_weights,ebd_weights);}
    data.load_data(infile);

    WSDGraph G;
    Proximity *prox = new Proximity();
    data.get_graph(G,prox);
    delete prox;
    if(nullifydiag){G.nullify_diag(0);}
    if(enn != -1){G.eps_nn(enn);}
    if(knn != -1){G.knn(knn);}

    G.label_propagation(0.0000001);
    wofstream outgraph((outdir+string("/wsd.json")).c_str());
    G.json(outgraph);
    outgraph.close();

    wofstream out_codes((outdir+string("/wsd.codes")).c_str());
    data.dump_Ycodes(out_codes);
    out_codes.close();
}

