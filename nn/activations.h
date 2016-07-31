#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "units.h"


class ReLUActivation : public LinearActivation{

public:
  ReLUActivation(unsigned dimensionality);
  void predict();
  void backprop();
};


class TanhActivation : public LinearActivation{

public:
  TanhActivation(unsigned dimensionality);
  void predict();
  void backprop();

};



#include "activations.hpp"
#endif
