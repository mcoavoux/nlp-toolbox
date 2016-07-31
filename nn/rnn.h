#ifndef RNN_H
#define RNN_H

#include <unordered_map>
#include <stdexcept>
#include <fstream>
#include "units.h"
#include "arrayfire.h"
#include "ffwd.h"



/*
 * This is a class for storing a batch of <Y,X> data sequences for BPTT
 */
template <class OUTSYM,class INSYM>
class SequentialMiniBatch{

public:

  SequentialMiniBatch(unsigned batch_sequence_length, vector<unsigned> const &types_sizes);
  SequentialMiniBatch(SequentialMiniBatch const &other);
  ~SequentialMiniBatch(){};

  //method for adding data to the minibatch
  virtual void add_sequenceXY(vector<vector<INSYM>> const &xvalues,vector<OUTSYM> const &yvalues);

  //method to finalize the minibatch before use returns true if minibatch is large enough; false if minibatch is too small for use
  virtual bool finalize();

  //methods for extracting data from the minibatch
  virtual const vector<OUTSYM>&   getYbatch(unsigned timestep) const {return yvector[timestep];}
  virtual const vector<INSYM>&    getXbatch(unsigned timestep,unsigned type_idx)const{return xvector[timestep][type_idx];}
  virtual const vector<vector<unsigned>>& get_eos() const {return eof_seq;}

  virtual unsigned get_batch_N()const{return yvector[0].size();}
  virtual unsigned get_batch_sequence_length()const{return seq_length;} //num time steps
  virtual unsigned get_num_types()const{return types_sizes.size();}

protected:

  virtual void pushYdata(unsigned timestep, OUTSYM const &ysymbol, bool eos);
  virtual void pushXdata(unsigned timestep, unsigned type_idx, vector<INSYM> const &xsymbols) { xvector[timestep][type_idx].insert(xvector[timestep][type_idx].end(),xsymbols.begin(),xsymbols.end());} 
  virtual void pushXdata(unsigned timestep, unsigned type_idx, INSYM const &xsymbol)  { xvector.at(timestep).at(type_idx).push_back(xsymbol);} 

  virtual void reset();
  virtual void xreset();
  virtual void yreset();
   
  vector<vector<OUTSYM>> yvector;        //[i] = timestep ; [j] = value
  vector<vector<vector<INSYM>>> xvector; //[i] = timestep ; [j] = type t ; [k] = value
  vector<vector<unsigned>> eof_seq;          //vector marking the end of observed sequences[i] = timestep ; [j] = value
  
  unsigned seq_length;//the length of a sequence within the batch
  vector<unsigned> types_sizes;   //num tokens per type
  int T = 0; //timestep
};


typedef vector<string> XTOKEN;
typedef vector<vector<string>> XSEQUENCE; 
typedef vector<string> YSEQUENCE;  
void read_string_dataset(istream *inFile,vector<XSEQUENCE> &xvalues,vector<YSEQUENCE> &yvalues);



template <class OUTSYM,class INSYM>  //all input layers must share the same C++ type (e.g. string or int)
class SymbolicRecurrentNetwork{

 public:
  SymbolicRecurrentNetwork(){};

  //network geometry and network building functions
  void add_layer(char const *name,NetworkUnit *layer);
  void add_input_layer(char const *name,InputSymbolicUnit<INSYM> *inlayer);
  //void add_recurrent_layer(char const *name,RecurrentLayer *reclayer);
  void set_output_layer(char const *name,TopSymbolicUnit<OUTSYM> *out_layer);
  void connect_layers(char const *nameA,char const *nameB); //parent -> child
  void connect_recurrent_layers(char const *source,char const *destination); //source output is fed as destination input at next timestep

  /* 
   * Method feeding data for processing next batch (note that a full data set is also a batch)
   * @param ydata : Y values, one subvector per sequence
   * @param xdata : 1st vector = 1st lookup unit, 2nd vector = 2nd lookup unit ... nth vector = nth lookup unit
   */
  void set_batch_data(vector<vector<vector<INSYM>>> const &xvalues, vector<vector<OUTSYM>> const &yvalues,unsigned batch_sequence_length);
  void set_batch_data(SequentialMiniBatch<OUTSYM,INSYM> const &batch);
  vector<unsigned> get_input_vocab_sizes()const;

  //TRAINING
  float train_one(float alpha,bool adagrad = true, bool average=true); //performs one pred->backprop->udpate cycle on the current batch and returns the unaveraged loss for this batch
  //may be useful for true online learning.

  /**
   * Performs an iterative truncated backprop through time on a sequential data set.
   * @param ydata           : Y values ; the size of this vector == train_size
   * @param xdata           : X values ; same conventions as set_batch_data.
   * @param epochs          : num iterations
   * @param mini_batch_size : the size of the minibatches used for training (<= overall batch size)
   * @param batch_sequence_length : internal length of the minibatch (length of the truncation)
   * @param alpha           : the overall learning rate
   * @param adagrad         : uses AdaGrad
   * @param average         : performs ASGD instead of SGD. The value tells the epoch where to start averaging
   * @return : the averaged loss for the last iteration.
   */
  void train_all(vector<vector<vector<INSYM>>> const &xdata,vector<vector<OUTSYM>> const &ydata,
		 unsigned epochs,
		 unsigned mini_batch_size,
		 unsigned batch_sequence_length,
		 float alpha,
		 bool adagrad = true,
		 int average  = -1);

  //PREDICTION AND EVALUATION
  //these evals may be slightly innacurate since we use truncated batches
  float eval(vector<vector<vector<INSYM>>> const &xvalues, vector<vector<OUTSYM>> const &yvalues,unsigned batch_sequence_length);
  //uses averaged weights instead of regular weights for evaluation
  float eval_avg(vector<vector<vector<INSYM>>> const &xvalues, vector<vector<OUTSYM>> const &yvalues,unsigned batch_sequence_length); 
 
  vector<OUTSYM> predict(vector<vector<INSYM>> const &xvalues);

protected:

  void compileNetwork();
  bool check_network_geometry(); //assumes topological order has been made (otherwise irrelevant).
  void predict_sequence();
  void backprop_sequence();

  void next_step();
  void prev_step();
  void reset_timestep();//resets the RNN to time step 0

  void update(float alpha,bool average,bool adagrad);
 
private:
  LayerMap names2layers;                       //maps layer names to the memory and memory manager for layers.
  vector<NetworkUnit*> topological_order;
  vector<InputSymbolicUnit<INSYM>*> input_layers;
  vector<pair<LinearActivation*,RecurrentLayer*>> recurrent_connections; //vector of recurrent layers taking their input from an internal memory
  TopSymbolicUnit<OUTSYM> *output_layer;
  bool is_compiled = false;
  bool has_output = false;
};

#include "rnn.hpp"
#endif
