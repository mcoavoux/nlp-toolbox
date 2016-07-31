#include "units.h"
#include "layers.h"
#include "activations.h"
#include "losses.h"
#include "lookup.h"
#include "rnn.h"
#include <sstream>

int main(){

  vector<unsigned> vocab_size = {1,2};
  string str("le m s D\n\
     chat m s N\n\
     dort - - V\n\
     sur - - P\n\
     le  m s D\n\
     paillasson  m s N\n\
     . - - PONCT\n\
\n\
     les f pl D\n\
     souris f pl N\n\
     dorment - - V\n\
     sur - - P\n\
     le  m s D\n\
     paillasson  m s N\n\
     . - - PONCT\n\
\n\
     le m s D\n\
     chat m s N");

  vector<XSEQUENCE> xvec;
  vector<YSEQUENCE> yvec;
  std::stringstream *infile = new std::stringstream(str);
  read_string_dataset(infile,xvec,yvec);
  delete infile;
  

  vector<string> outsyms     = {"D","N","V","P","PONCT"};
  vector<string> input_symsA = {"le","chat","dort","sur","paillasson","les","souris","dorment","."};
  vector<string> input_symsB = {"m","s","f","pl","-"};

  SymbolicRecurrentNetwork<string,string> net;
  net.set_output_layer("loss",new SoftMaxLoss<string>(outsyms));
  net.add_layer("top",new LinearLayer()); 
  net.add_layer("hidden",new ReLUActivation(10));
  net.add_layer("A",new LinearLayer());
  net.add_layer("B",new LinearLayer());
  net.add_layer("C",new RecurrentLayer());
  net.add_input_layer("lookupA", new LinearLookup<string>(input_symsA,1,12)); //this layer consumes 2 token for each example processed and builds embeddings of size 10
  net.add_input_layer("lookupB", new LinearLookup<string>(input_symsB,2,6));  //this layer consumes 1 token for each example processed and builds embeddings of size 10

  net.connect_layers("loss","top");
  net.connect_layers("top","hidden");
  net.connect_layers("hidden","A");
  net.connect_layers("hidden","B");
  net.connect_layers("hidden","C");
  net.connect_layers("A","lookupA");
  net.connect_layers("B","lookupB");
  net.connect_recurrent_layers("hidden","C");

  /* net.set_batch_data(xvec,yvec,7);
  for(int i = 0; i < 100;++i){
    cout << "loss = " <<  net.train_one(0.02,false,true) << endl; 
  }
  */
  net.train_all(xvec,yvec,100,3,7,0.02,true,50); //
  
  cout << "accurracy: " << net.eval_avg(xvec,yvec,7) << endl;

  return 0;
}
