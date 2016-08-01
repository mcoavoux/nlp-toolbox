#ifndef UNITS_H
#define UNITS_H

#include "arrayfire.h"
#include <vector>
#include <string>
#include <stdexcept>

using namespace std;

/*
 * This module specifies an abstract interface for neural network units.
 */


class UnimplementedMethodException: public std::runtime_error{
public:
  UnimplementedMethodException(string methodname):runtime_error((string("This method cannot be implemented : ") + methodname + "there is likely an error in the network architecture")){}
};

class InvalidNetworkException: public std::runtime_error{
public:
 InvalidNetworkException():runtime_error(string("The network geometry makes no sense.")){}
};


/**
 * This abstract class is the base unit of an inner DAG computation structure.
 * The network ensures that forward and backward computations are run in the correct order
 */
class NetworkUnit{

public:  

  virtual ~NetworkUnit(){} 

  //Model
  virtual void predict()  = 0;
  virtual void backprop() = 0;
  virtual void update(float alpha,bool average,bool adagrad) = 0;
 
  virtual af::array& get_predictions() = 0;
  virtual af::array& get_delta() = 0;

  virtual unsigned in_dimensionality() const = 0;
  virtual unsigned out_dimensionality()const = 0;

  //Network topology
  virtual NetworkUnit* get_child_at(unsigned idx)  const{return children[idx];}
  virtual NetworkUnit* get_parent_at(unsigned idx) const{return parents[idx];}
  virtual void add_child(NetworkUnit *child) = 0;         
  virtual void add_parent(NetworkUnit *parent) = 0;      
  virtual size_t children_size() const{return children.size();}
  virtual size_t parent_size()   const{return parents.size();}
  virtual void set_name(string const &name){logical_name = name;};
  virtual string& get_name(){return logical_name;};

  //averaging
  virtual void swap_weights() = 0;//switches averaged and regular weights.

  //time sequence (for recurrent networks)
  virtual void set_seq_length(unsigned L) = 0;
  virtual unsigned get_seq_length() = 0;

  virtual bool has_next_timestep(){return ((T) < get_seq_length());};
  virtual bool has_prev_timestep(){return T >=0;};

  virtual int get_timestep()const{return T;};
  virtual void next_timestep(){++T;};
  virtual void prev_timestep(){--T;};
  virtual void reset_timestep(){T=0;};

protected:
  string logical_name;
  vector<NetworkUnit*> children;
  vector<NetworkUnit*> parents;
  int T; //time step

};


class RecurrentLayer;
class LinearActivation : public NetworkUnit{

public:
  
  LinearActivation(unsigned dimensionality);
  virtual ~LinearActivation(){} 
  virtual void predict();
  virtual void backprop();
  virtual void update(float alpha,bool average,bool adagrad){}; //these units have no weight

  virtual af::array& get_predictions() {return predictions[T];};
  virtual af::array& get_delta()       {return delta[T];};

  virtual void add_child(NetworkUnit *child)   {children.push_back(child);};         
  virtual void add_parent(NetworkUnit *parent) {parents.push_back(parent);};    
  virtual void set_recurrent_parent(RecurrentLayer *parent) {rec_parent = parent; has_rec_parent = true;};   

  virtual unsigned in_dimensionality() const{return dimensionality;}; //dimension of the activation 
  virtual unsigned out_dimensionality()const{return dimensionality;}; //dimension of the activation 
  virtual void swap_weights(){}; //these units have no params, do nothing

  virtual void set_seq_length(unsigned L);
  virtual unsigned get_seq_length();
  


 protected:  
  void init_stack();
  virtual void gather_input();
  virtual void gather_delta();
  
  vector<af::array> predictions;
  vector<af::array> delta;
  unsigned dimensionality;
  
  //rnn
  bool has_rec_parent = false;
  RecurrentLayer *rec_parent;
};




/**
 * This abstract class is an abstract top level class for the loss of a network
 */
template <class SYMTYPE>
class TopSymbolicUnit : public NetworkUnit{

 public:
  virtual ~TopSymbolicUnit(){} 
  virtual void predict()         = 0;
  virtual void backprop()        = 0;
  virtual void update(float alpha,bool average,bool adagrad) {}; //these units have no param -> this method allows for generic polymorphic implementations

  virtual float get_loss() const = 0;
  virtual void set_reference(vector<SYMTYPE> const &yreference) = 0;//reference against which we compute the loss, backpropagate etc.

  virtual vector<SYMTYPE>& extract_argmaxes() = 0; //extracts the vector of argmaxes for this batch
  virtual af::array& get_predictions() = 0;
  virtual af::array& get_delta() = 0;

  virtual unsigned in_dimensionality()const = 0;  //dimension of the output vector
  virtual unsigned out_dimensionality()const = 0; //dimension of the output vector

  virtual NetworkUnit* get_parent_at(unsigned idx) const{throw UnimplementedMethodException(string("get_parent_at"));}
  virtual void add_parent(NetworkUnit *parent)          {throw UnimplementedMethodException(string("add_parent"));}
  virtual void add_child(NetworkUnit *child)            {children.push_back(child);};  

  //averaging
  virtual void swap_weights(){};//these units have no params, do nothing
  
};

/**
 * This abstract class is an abstract input level class for symbolic network
 */
template <class SYMTYPE>
class InputSymbolicUnit : public NetworkUnit{

 public:  
  virtual ~InputSymbolicUnit(){} 
  virtual void predict()         = 0;
  virtual void backprop()        = 0;
  virtual void update(float alpha,bool average,bool adagrad) = 0;
  virtual void set_input(vector<SYMTYPE> const &xinput) = 0;//x symbolic data to be propagated through the network

  virtual af::array& get_predictions() = 0;
  virtual af::array& get_delta() {throw UnimplementedMethodException(string("get_child_at"));};

  virtual unsigned in_dimensionality()  const = 0;//dimension of the encoded input vector
  virtual unsigned out_dimensionality() const = 0;//dimension of the encoded input vector
  virtual unsigned vocab_size()const = 0; //number of discrete symbols expected as input

  virtual NetworkUnit* get_child_at(unsigned idx)  const{throw UnimplementedMethodException(string("get_child_at"));}
  virtual void add_child(NetworkUnit *child)            {throw UnimplementedMethodException(string("get_child_at"));}
  virtual void add_parent(NetworkUnit *parent) = 0;      

  //averaging
  virtual void swap_weights() = 0;//switches averaged and regular weights.  

};

#include "units.hpp"
#endif
