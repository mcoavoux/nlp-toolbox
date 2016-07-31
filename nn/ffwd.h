#ifndef FFWD_H
#define FFWD_H

#include <unordered_map>
#include <stdexcept>
#include "units.h"

class InvalidArchException: public std::runtime_error{
public:
  InvalidArchException(string token):runtime_error((string("Invalid Arch exception :") + token)){}
};


class LayerMap{
  /*
   * Maps Layer logical names to their pointer and stores layers (memory manager for layers)
   */
 public:
  LayerMap();
  ~LayerMap();
  bool has_unitname(string const &name)const;
  void add_unit(string const &name,NetworkUnit *layer);
  NetworkUnit* get_layer(string const &name);
  NetworkUnit* operator()(string const &name);
  unsigned size()const{return name2mem.size();}
 private: 
  unordered_map<string,NetworkUnit*> name2mem;
};

/*
* General purpose computation graph for arbitrary feed forward networks (but not recurrent ones)
*/
template <class OUTSYM,class INSYM>  //all input layers must share the same C++ type (e.g. string or int) 
class SymbolicFeedForwardNetwork{

 public:
  SymbolicFeedForwardNetwork(){};

  //network geometry and network building functions
  void add_layer(char const *name,NetworkUnit *layer);
  void add_input_layer(char const *name,InputSymbolicUnit<INSYM> *inlayer);
  void set_output_layer(char const *name,TopSymbolicUnit<OUTSYM> *out_layer);
  void connect_layers(char const *nameA,char const *nameB);//parent -> child

  /* Method feeding data for processing next batch (note that a full data set is also a batch)
   * @param ydata : Y values ; the size of this vector == batch_size
   * @param xdata : 1st vector = 1st lookup unit, 2nd vector = 2nd lookup unit ... nth vector = nth lookup unit 
                    moves to next example when lookup unit vocab_size() symbols have been processed.
		    for a lookup unit i the ith vector should contain batch_size * vocab_size() symbols
  */
  void set_batch_data(vector<OUTSYM> const &ydata,
		      vector<vector<INSYM>> const &xdata);
 
  /* As above but for prediction only, hence no Y data */
  void set_batch_data(vector<vector<INSYM>> const &xdata); 

  //TRAINING
  float train_one(float alpha,bool adagrad = true, bool average=true); //performs one pred->backprop->udpate cycle on the current batch and returns the unaveraged loss for this batch
                                //may be useful for true online learning.

  /** 
   * Performs iterative training on the current data batch.
   * @param ydata           : Y values ; the size of this vector == train_size
   * @param xdata           : X values ; same conventions as set_batch_data.
   * @param epochs          : num iterations
   * @param mini_batch_size : the size of the minibatches used for training (<= overall batch size)
   * @param alpha           : the overall learning rate
   * @param adagrad         : uses adagrad
   * @param average         : performs ASGD instead of SGD. The value tells the epoch where to start averaging
   * @return : the averaged loss for the last iteration
   */
  void train_all(vector<OUTSYM> const &ydata,
		 vector<vector<INSYM>> const &xdata,
		 unsigned epochs,
		 unsigned mini_batch_size,
		 float alpha,
		 bool adagrad = true,
		 int average  = -1);

  void train_all(DataSampler<OUTSYM,INSYM> sampler,
		 unsigned epochs,
		 unsigned mini_batch_size,
		 float alpha,
		 bool adagrad = true,
		 int average  = -1);

  //PREDICTION AND EVALUATION  
  float eval(vector<OUTSYM> const &ydata,
	     vector<vector<INSYM>> const &xdata); 
  float eval_avg(vector<OUTSYM> const &ydata,
		 vector<vector<INSYM>> const &xdata); //uses averaged weights instead of regular weights for evaluation
 
  vector<OUTSYM>& predict(vector<vector<INSYM>> const &xdata);
  OUTSYM predict_one(vector<vector<INSYM>> const &xdata);//supposes the xdata dimensions is that of a unique example. crashes otherwise (!)

protected:

  void compileNetwork();
  bool check_network_geometry(); //assumes topological order has been made (otherwise irrelevant).

  void predict_batch(); 
  void backprop_batch();
  void update_batch(float alpha,bool average,bool adagrad);
 
 private:
  LayerMap names2layers;                       //maps layer names to the memory and memory manager for layers.
  vector<NetworkUnit*> topological_order;
  vector<InputSymbolicUnit<INSYM>*> input_layers;
  TopSymbolicUnit<OUTSYM> *output_layer;
  bool is_compiled = false;
  bool has_output = false;
};

#include "ffwd.hpp"
#endif


