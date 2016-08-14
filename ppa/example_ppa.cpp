#include "units.h"
#include "layers.h"
#include "activations.h"
#include "losses.h"
#include "lookup.h"
#include "ffwd.h"
#include "ppa.h"
#include <omp.h>

int run_sampler(unsigned epochs,float alpha,unsigned batch_size){

    //load sampler
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

    //load dev and test
    PPADataEncoder dev_set("PPAttachData/devset.lemma");
    PPADataEncoder test_set("PPAttachData/test.lemma");

    //load Word vectors 
    Word2vec w2v;
    vector<string> wvdict;
    af::array w2v_embeddings;
    w2v.load_dictionary("PPAttachData/embeddings/deps.words.lemmatized");
    //w2v.filter(xdict);

    //make network
    vector<string> ydict;
    samp.getYdictionary(ydict);
    SymbolicFeedForwardNetwork<string,string> net;
    net.set_output_layer("loss",new SoftMaxLoss<string>(ydict));
    net.add_layer("top",new LinearLayer());  
    net.add_layer("hidden",new ReLUActivation(100));
    net.add_layer("A",new LinearLayer());
    net.add_input_layer("lookupA",new LinearLookup<string>(w2v.get_keys(),w2v.get_values(),4,false));
    net.connect_layers("loss","top");
    net.connect_layers("top","hidden");
    net.connect_layers("hidden","A");
    net.connect_layers("A","lookupA");

    for(int E = 0; E < epochs;++E){
      vector<string> ydata;
      vector<vector<string>> xdata;
      //af::timer start1 = af::timer::start();
      samp.generate_sample(ydata,xdata,batch_size);
      //printf("elapsed seconds (sampling): %g\n", af::timer::stop(start1));

      PPADataEncoder sampdata(ydata,xdata);
      vector<string> enc_ydata;
      vector<vector<string>> enc_xdata(1,vector<string>());
      sampdata.getYdata(enc_ydata);
      sampdata.getXdata(enc_xdata[0]);

      //af::timer start2 = af::timer::start();
      net.set_batch_data(enc_ydata,enc_xdata);
      float loss = net.train_one(alpha,true,true);
      //printf("elapsed seconds (backprop): %g\n", af::timer::stop(start2));


      if (E % 20 == 0){
	vector<string> devy;
	vector<vector<string>> devx(1,vector<string>());
	dev_set.getYdata(devy);	
	dev_set.getXdata(devx[0]);	
	float acc = net.eval_avg(devy,devx);        //auto-eval on dev data
        cout << "epoch " << E << ", loss= " << loss << ", eval (dev) = " << acc << endl;
      }else {
	cout << "epoch" << E <<endl;
      }
	      
    }
    vector<string> testy;
    vector<vector<string>> testx(1,vector<string>());
    test_set.getYdata(testy);	
    test_set.getXdata(testx[0]);	
    float acc = net.eval_avg(testy,testx); 
    cout << "final eval (test) = " << acc << endl;
    return 0;
}

  /*
  vector<string> yvalues;
  vector<vector<string>> xvalues;
  samp.generate_sample(yvalues,xvalues,100000);
  samp.dump_sample(cout,yvalues,xvalues);
  */

int run_vanilla(unsigned epochs, float alpha){

 //Data set (Ratnaparkhi's 94 RRR data set)
  vector<string> ydict;
  vector<string> xdict;
  PPADataEncoder data_set("PPAttachData/training.lemma");
  data_set.add_data("PPAttachData/devset.lemma");
  data_set.add_data("PPAttachData/test.lemma");
  data_set.getYdictionary(ydict);
  data_set.getXdictionary(xdict);

  //External Word vectors 
  Word2vec w2v;
  vector<string> wvdict;
  af::array w2v_embeddings;
  w2v.load_dictionary("PPAttachData/embeddings/deps.words.lemmatized");
  w2v.filter(xdict);

  //training set
  data_set.clear();
  data_set.add_data("PPAttachData/training.lemma");
  data_set.getXdictionary(xdict);
  
  //build network
  SymbolicFeedForwardNetwork<string,string> net;
  net.set_output_layer("loss",new SoftMaxLoss<string>(ydict));
  net.add_layer("top",new LinearLayer());  
  net.add_layer("hidden",new ReLUActivation(400));
  net.add_layer("A",new LinearLayer());
  net.add_input_layer("lookupA",new LinearLookup<string>(w2v.get_keys(),w2v.get_values(),data_set.x_vocab_size(),true));
  net.connect_layers("loss","top");
  net.connect_layers("top","hidden");
  net.connect_layers("hidden","A");
  net.connect_layers("A","lookupA");

  vector<string> ydata;
  vector<vector<string>> xdata(1,vector<string>());
  data_set.getYdata(ydata);
  data_set.getXdata(xdata[0]);
  net.set_batch_data(ydata,xdata);
  net.train_all(ydata,xdata,epochs,100,alpha,true,epochs/2);//10 epochs, batch size= 100,alpha=0.01, Adagrad=On, start averaging at 5th epoch

  PPADataEncoder dev_set("PPAttachData/devset.lemma");
  dev_set.getYdata(ydata);
  dev_set.getXdata(xdata[0]);
  float dev_acc = net.eval_avg(ydata,xdata);   
  cout << "dev acc = " << dev_acc << endl; 

  PPADataEncoder test_set("PPAttachData/test.lemma");  
  test_set.getYdata(ydata);
  test_set.getXdata(xdata[0]);
  float test_acc = net.eval_avg(ydata,xdata);   
  cout << "test acc = " << test_acc << endl; 
  return 0;
}

int run_vanilla_nolemma(unsigned epochs, float alpha){

 //Data set (Ratnaparkhi's 94 RRR data set)
  vector<string> ydict;
  vector<string> xdict;
  PPADataEncoder data_set("PPAttachData/training");
  data_set.add_data("PPAttachData/devset");
  data_set.add_data("PPAttachData/test");
  data_set.getYdictionary(ydict);
  data_set.getXdictionary(xdict);

  //External Word vectors 
  Word2vec w2v;
  vector<string> wvdict;
  af::array w2v_embeddings;
  w2v.load_dictionary("PPAttachData/embeddings/deps.words");
  w2v.filter(xdict);

  //training set
  data_set.clear();
  data_set.add_data("PPAttachData/training");
  data_set.getXdictionary(xdict);
  
  //build network
  cerr << "building network"<<endl;
  SymbolicFeedForwardNetwork<string,string> net;
  net.set_output_layer("loss",new SoftMaxLoss<string>(ydict));
  net.add_layer("top",new LinearLayer());  
  net.add_layer("hidden",new ReLUActivation(400));
  net.add_layer("A",new LinearLayer());
  net.add_input_layer("lookupA",new LinearLookup<string>(w2v.get_keys(),w2v.get_values(),data_set.x_vocab_size(),true));
  net.connect_layers("loss","top");
  net.connect_layers("top","hidden");
  net.connect_layers("hidden","A");
  net.connect_layers("A","lookupA");

  vector<string> ydata;
  vector<vector<string>> xdata(1,vector<string>());
  data_set.getYdata(ydata);
  data_set.getXdata(xdata[0]);
  net.set_batch_data(ydata,xdata);
  net.train_all(ydata,xdata,epochs,100,alpha,true,epochs/2);//10 epochs, batch size= 100,alpha=0.01, Adagrad=On, start averaging at 50th epoch

  PPADataEncoder dev_set("PPAttachData/devset");
  dev_set.getYdata(ydata);
  dev_set.getXdata(xdata[0]);
  float dev_acc = net.eval_avg(ydata,xdata);   
  cout << "dev acc = " << dev_acc << endl; 

  PPADataEncoder test_set("PPAttachData/test");  
  test_set.getYdata(ydata);
  test_set.getXdata(xdata[0]);
  float test_acc = net.eval_avg(ydata,xdata);   
  cout << "test acc = " << test_acc << endl; 
  return 0;
}



int main(){
  omp_set_num_threads(8);
  //run_vanilla_nolemma(400,0.05);
  //run_vanilla(800,0.05);
  run_sampler(10000,0.01,100);
} 
 

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

