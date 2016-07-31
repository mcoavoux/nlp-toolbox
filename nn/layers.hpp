#include "nnutils.h"
#include <limits>

LinearLayer::LinearLayer(){
  OutSize = 0;
  InSize  = 0;
}

void LinearLayer::init_stack(){
 
  nn_init_weights(Weights,OutSize,InSize);
  nn_init_weights(bias,OutSize);
  Jacobian = af::constant(0,OutSize,InSize);
  Jbias    = af::constant(0,OutSize);
  ypredictions.push_back(af::constant(0,OutSize));
  xdelta.push_back(af::constant(0,InSize));
  T = 0;  
}

void LinearLayer::add_child(NetworkUnit *child){
  if (!children.empty()){throw InvalidNetworkException();}
  children.push_back(child);
  InSize = child->out_dimensionality();
  if(OutSize != 0){init_stack();}
}         

void LinearLayer::add_parent(NetworkUnit *parent){
if (!children.empty()){throw InvalidNetworkException();}
  parents.push_back(parent);
  OutSize = parent->in_dimensionality();
  if(InSize != 0){init_stack();}
} 


void LinearLayer::set_seq_length(unsigned L){
  ypredictions.resize(L);
  xdelta.resize(L);
}

unsigned LinearLayer::get_seq_length(){
  return ypredictions.size();
}

void LinearLayer::predict(){

  unsigned batch_size = children[0]->get_predictions().dims(1);
  ypredictions[T]  = af::matmul(Weights,children[0]->get_predictions()) + af::tile(bias,1,batch_size);

}

void LinearLayer::backprop(){

  Jacobian  += af::matmulNT(parents[0]->get_delta(),children[0]->get_predictions());
  Jbias     += af::sum(parents[0]->get_delta(), 1);
  xdelta[T] = af::matmulTN(Weights,parents[0]->get_delta());     

}

void LinearLayer::update(float alpha,bool average,bool adagrad){
  
  if (adagrad){                          //ADAGRAD AND LEARNING RATE

    if (AdaGrad.isempty()){
      AdaGrad = af::pow2(af::constant(std::numeric_limits<float>::min(),Weights.dims(0),Weights.dims(1)));
      Adabias = af::pow2(af::constant(std::numeric_limits<float>::min(),bias.dims(0)));
    }

    Weights  +=  alpha * (Jacobian / af::sqrt(AdaGrad));
    bias     +=  alpha * (Jbias /  af::sqrt(Adabias));
    AdaGrad += af::pow2(Jacobian);
    Adabias += af::pow2(Jbias);  

  }else{  //apply regular learning rate
    Weights += alpha*Jacobian;
    bias    += alpha*Jbias;
  }

  if (average){                         //MOVING AVERAGE UPDATE

    if (AvgWeights.isempty()){
      AvgWeights = af::constant(0,Weights.dims(0),Weights.dims(1));
      Avgbias = af::constant(0,bias.dims(0));
      avgN = 0;
    }
    //as cumulative moving average
    AvgWeights = AvgWeights + (Weights - AvgWeights)/(avgN+1);
    Avgbias    = Avgbias + (bias - Avgbias)/(avgN+1); 
    ++avgN;
  }
  
  Jacobian = af::constant(0,OutSize,InSize);         //RESET
  Jbias    = af::constant(0,OutSize);
}

void LinearLayer::swap_weights(){

  if ( avgN > 0 ) {

    af::array tmpWeights = Weights;
    Weights = AvgWeights;
    AvgWeights = tmpWeights;
    
    af::array tmpbias = bias;
    bias              = Avgbias;
    Avgbias           = tmpbias;

  }
}

RecurrentLayer::RecurrentLayer(){
  this->OutSize = 0;
  this->InSize = 0;
}

void RecurrentLayer::set_memory_dimension(unsigned mem_dims){
  this->OutSize = mem_dims;
  this->InSize = mem_dims;
  init_stack();
}

void RecurrentLayer::set_seq_length(unsigned L){
  ypredictions.resize(L);
  xdelta.resize(L);
  xinput.resize(L);
}


void RecurrentLayer::set_next_input(af::array &xinput){
 
  if (T+1 < this->xinput.size()){
    this->xinput[T+1] = xinput;
    if (T < eof_sequences.size() && !eof_sequences[T].empty()){//mask for end of sequences
      af::array IDXES(eof_sequences[T].size(),eof_sequences[T].data());
      this->xinput[T+1](af::span,IDXES) = 0.0;  
    }
  }
}

af::array& RecurrentLayer::get_next_delta(){ //returns the delta from next time step
  return xdelta[T+1];
}


void RecurrentLayer::predict(){
  
  unsigned batch_size =xinput[T].dims(1);
  ypredictions[T]  = af::matmul(Weights,xinput[T]) + af::tile(bias,1,batch_size);

}

void RecurrentLayer::backprop(){

  Jacobian  += af::matmulNT(parents[0]->get_delta(),xinput[T]);
  Jbias     += af::sum(parents[0]->get_delta(), 1);
  xdelta[T] = af::matmulTN(Weights,parents[0]->get_delta());     

}

 

