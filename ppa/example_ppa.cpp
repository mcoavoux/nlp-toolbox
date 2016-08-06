#include "units.h"
#include "layers.h"
#include "activations.h"
#include "losses.h"
#include "lookup.h"
#include "ffwd.h"
#include "ppa.h"

int main(){

  string training_path = "PPAttachData/training.lemma";
  string param_path = "PPAttachData/wordsketches/";
  string vpath = param_path + string("vdistrib");
  string x1vpath = param_path + string("x1givenv");
  string pvpath = param_path + string("pgivenv");
  string x2vppath = param_path + string("x2givenvp");
  string px1path = param_path + string("pgivenx1");
  string x2x1ppath = param_path + string("x2givenx1p");
  
  DataSampler samp(training_path.c_str(),
		   vpath.c_str(),
		   x1vpath.c_str(),
		   pvpath.c_str(),
		   x2vppath.c_str(),
		   px1path.c_str(),
		   x2x1ppath.c_str());

  vector<string> yvalues;
  vector<vector<string>> xvalues;

  samp.generate_sample(yvalues,xvalues,10000);
  samp.dump_sample(cout,yvalues,xvalues);

  //sampler.getXdictionary(xdict);
  /*
vector<string> yvalues;
  vector<vector<string>> xvalues;
  sdata_set.original_data_set(yvalues,xvalues);
  for(int i = 0 ; i < 100;++i){
    sdata_set.sample_example(yvalues,xvalues,0);
  }
  exit(1);
  */ 

  //Data set (Ratnaparkhi's 94 RRR data set)
  /* 
 vector<string> ydict;
  vector<string> xdict;
  PPADataEncoder data_set("PPAttachData/training");
  data_set.add_data("PPAttachData/devset");
  data_set.add_data("PPAttachData/test");
  data_set.getYdictionary(ydict);
  data_set.getXdictionary(xdict);
  
  */

  //External Word vectors 
  //Word2vec w2v;
  //vector<string> wvdict;
  //af::array w2v_embeddings;
  //w2v.load_dictionary("../w2v/deps.words");
  //w2v.filter(xdict);

  //training set
  /*
    data_set.clear();
  data_set.add_data("PPAttachData/training");
  data_set.getXdictionary(xdict);
  */

  //build network
  /*
  SymbolicFeedForwardNetwork<string,string> net;
  net.set_output_layer("loss",new SoftMaxLoss<string>(ydict));
  net.add_layer("top",new LinearLayer());  
  net.add_layer("hidden",new ReLUActivation(100));
  net.add_layer("A",new LinearLayer());
  net.add_input_layer("lookupA",new LinearLookup<string>(w2v.get_keys(),w2v.get_values(),data_set.x_vocab_size(),true));
  //net.add_input_layer("lookupA",new LinearLookup<string>(xdict,data_set.x_vocab_size(),50,string("???")));//last param provides an additional symbol for unk words 
  net.connect_layers("loss","top");
  net.connect_layers("top","hidden");
  net.connect_layers("hidden","A");
  net.connect_layers("A","lookupA");

  //train 
  for(int E = 0; E < 100;++E){
    vector<string> ydata;
    vector<vector<string>> xdata;
    sampler.sample_data_set(ydata,xdata);
    PPADataEncoder sdata(ydata,xdata);
    sdata.getYdata(ydata);
    sdata.getXdata(xdata[0]);
    net.set_batch_data(ydata,xdata);
    net.train_all(ydata,xdata,10,100,0.01,true,5);//10 epochs, batch size= 100,alpha=0.01, Adagrad=On, start averaging at 5th epoch
    float acc = net.eval_avg(ydata,xdata);        //auto-eval on train data
    cout << "eval = " << acc << endl;
  }
 
  // data_set.getYdata(ydata);
  // data_set.getXdata(xdata[0]);
  // net.set_batch_data(ydata,xdata);
  // net.train_all(ydata,xdata,10,100,0.01,true,5);//10 epochs, batch size= 100,alpha=0.01, Adagrad=On, start averaging at 5th epoch
  // float acc = net.eval_avg(ydata,xdata);        //auto-eval on train data
  // cout << "eval = " << acc << endl;
  cout << "test" << endl;
  vector<string> ydata;
  vector<vector<string>> xdata;
  PPADataEncoder test_set("PPAttachData/test");
  test_set.getYdata(ydata);
  test_set.getXdata(xdata[0]);
  float test_acc = net.eval_avg(ydata,xdata);        //eval on test data
  cout << "eval = " << test_acc << endl;
  */
}
