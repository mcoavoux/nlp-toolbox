#include "units.h"
#include "layers.h"

LinearActivation::LinearActivation(unsigned dimensionality){
  this->dimensionality = dimensionality;
  init_stack();
}

void LinearActivation::predict(){
  gather_input();
}

void LinearActivation::backprop(){
  gather_delta();
}

void LinearActivation::init_stack(){
  predictions.push_back(af::constant(0,dimensionality));
  delta.push_back(af::constant(0,dimensionality));
  T = 0;
}

void LinearActivation::set_seq_length(unsigned L){
  predictions.resize(L);  
  delta.resize(L);  
}

unsigned LinearActivation::get_seq_length(){
  return predictions.size();
}

void LinearActivation::gather_input(){        
  predictions[T] =  children[0]->get_predictions();
  for(int i = 1; i < children.size();++i){
    predictions[T] += children[i]->get_predictions();
  }
}

void LinearActivation::gather_delta(){   
     
  delta[T] =  parents[0]->get_delta();
  for(int i = 1; i < parents.size();++i){
    delta[T] += parents[i]->get_delta();
  }
  if (has_rec_parent && ((T+1) < delta.size())){delta[T] += rec_parent->get_next_delta();}
}
