#ifndef RLAYER_H
#define RLAYER_H

#include "arrayfire.h"
#include<stack>


class RecurrentLinearLayer{
  
public:

  RecurrentLinearLayer(unsigned out_size,unsigned in_size,unsigned memory_size);

  virtual void predict();
  virtual void backprop();
  virtual void update(float alpha,bool average,bool adagrad);

  unsigned in_dimensionality()const{return InSize;};
  unsigned out_dimensionality()const{return OutSize;};
  
  virtual af::array& get_predictions(){return ypredictions.top();}
  virtual af::array& get_delta(){return xdelta.top();}

  virtual void swap_weights();
  
  //for folding/unfolding the network
  virtual void push(af::array &xinput);
  virtual void pop();

protected:

  void gather_xinput();
  void gather_ydelta();

  unsigned OutSize, InSize;
  std::stack<af::array> ypredictions, ydelta;
  std::stack<af::array> xmemory,xinput,xdelta;
  af::array MWeights, MJacobian,Mbias,MJbias;
  af::array Weights,  Jacobian, bias, Jbias;
  af::array AvgMWeights,AvgMbias,AvgWeights,Avgbias;
  float avgN = 0;//counts the number of updates when averaging
  af::array MAdaGrad,MAdabias,AdaGrad,Adabias;
};


#include "rlayers.hpp"
#endif
