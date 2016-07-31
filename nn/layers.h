#ifndef LAYER_H
#define LAYER_H

#include "arrayfire.h"
#include "units.h"
#include <stack>

using namespace std;

class LinearLayer : public NetworkUnit{
  
public:

  LinearLayer();

  virtual void predict();
  virtual void backprop();
  virtual void update(float alpha,bool average,bool adagrad);

  unsigned in_dimensionality()const{return InSize;};
  unsigned out_dimensionality()const{return OutSize;};

  virtual void add_child(NetworkUnit *child);         
  virtual void add_parent(NetworkUnit *parent); 
  
  virtual af::array& get_predictions(){return ypredictions[T];}
  virtual af::array& get_delta(){return xdelta[T];}

  virtual void swap_weights();

  virtual void set_seq_length(unsigned L);
  virtual unsigned get_seq_length();

protected:

  void init_stack();
  unsigned OutSize,InSize;
  vector<af::array> ypredictions;
  vector<af::array> xdelta;
  af::array Weights, Jacobian;
  af::array bias,Jbias;
  af::array AvgWeights,Avgbias;
  float avgN = 0;//counts the number of updates when averaging
  af::array AdaGrad,Adabias;
};

class RecurrentLayer : public LinearLayer{

public:

  RecurrentLayer();
  virtual void predict();
  virtual void backprop();
  virtual void init_input(unsigned batch_size){this->xinput.at(0) = af::constant(0,this->InSize,batch_size);};//sets the input for timestep 0 to 0
  virtual void set_memory_dimension(unsigned mem_dims);
  virtual void set_next_input(af::array &xinput);//sets the input for next timestep (used by recurrent connections)
  virtual af::array& get_next_delta(); //returns the delta from next time step
  virtual void set_seq_length(unsigned L);
  virtual void set_eos(vector<vector<unsigned>> const &eof_sequences){this->eof_sequences = eof_sequences;}    

protected:

  vector<af::array> xinput; //this is the recurrent input vector
  vector<vector<unsigned>> eof_sequences;//vector indicating which sequences end at which timestep for a batch position
};

#include "layers.hpp"
#endif


