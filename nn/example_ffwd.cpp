#include "units.h"
#include "layers.h"
#include "activations.h"
#include "losses.h"
#include "lookup.h"
#include "ffwd.h"


int main(){
  //af::setDevice(0);
    
  vector<string> outsyms = {"A","B"};
  vector<string> input_symsA = {"x","y","z","w"};
  vector<string> input_symsB = {"a","b","c","d"};

  //build network
  SymbolicFeedForwardNetwork<string,string> net;
  net.set_output_layer("loss",new SoftMaxLoss<string>(outsyms));
  net.add_layer("top",new LinearLayer()); 
  net.add_layer("hidden",new ReLUActivation(4));
  net.add_layer("A",new LinearLayer());
  net.add_layer("B",new LinearLayer());
  net.add_input_layer("lookupA",new LinearLookup<string>(input_symsA,2,5)); //this layer consumes 2 token for each example processed and builds embeddings of size 5
  net.add_input_layer("lookupB",new LinearLookup<string>(input_symsB,1,5)); //this layer consumes 1 token for each example processed and builds embeddings of size 5
  net.connect_layers("loss","top");
  net.connect_layers("top","hidden");
  net.connect_layers("hidden","A");
  net.connect_layers("hidden","B");
  net.connect_layers("A","lookupA");
  net.connect_layers("B","lookupB");

  //data set
  vector<string> ydata  = {"A","A","B","B"};
  vector<string> xdataA = {"x","w","x","y","w","y","x","z"};
  vector<string> xdataB = {"a","a","b","a"};
  vector<vector<string>> xdata;
  xdata.push_back(xdataA);
  xdata.push_back(xdataB);

  //train
  net.set_batch_data(ydata,xdata);
  net.train_all(ydata,xdata,300,4,0.01,false,50);//100 epochs, batch size= 4, alpha=0.01,Adagrad=On, start averaging at 50th epoch
  float acc = net.eval_avg(ydata,xdata);
  cout << "eval = " << acc << endl;
  return 0;
}
