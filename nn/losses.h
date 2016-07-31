#ifndef LOSSES_H
#define LOSSES_H


#include "arrayfire.h"
#include "units.h"
#include "nnutils.h"

using namespace std;

template <class SYMTYPE>
class SoftMaxLoss : public TopSymbolicUnit<SYMTYPE>{
  
public:

  SoftMaxLoss(vector<SYMTYPE> const &dict);

  void predict();
  void backprop();
  float get_loss()const;
  void set_reference(vector<SYMTYPE> const &yreference);

  unsigned in_dimensionality()const{return Size;};
  unsigned out_dimensionality()const{return Size;};
 

  vector<SYMTYPE>& extract_argmaxes();
  af::array& get_predictions(){return ypredictions[this->T];};
  af::array& get_delta(){return xdelta[this->T];};

  virtual void set_seq_length(unsigned L);
  virtual unsigned get_seq_length();

protected:
  void init_stack();
  void gather_xinput();

private:
  unsigned Size;//dimensionality
  vector<af::array> yreference,ypredictions;
  vector<af::array> xinput,xdelta;
  BiMap<SYMTYPE> ymapper;
  vector<SYMTYPE> argmaxes;
  //unsigned T;//time step
};


#include "losses.hpp"
#endif
