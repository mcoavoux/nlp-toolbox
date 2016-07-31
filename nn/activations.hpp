ReLUActivation::ReLUActivation(unsigned dimensionality) : LinearActivation(dimensionality){}
 
void ReLUActivation::predict(){
  gather_input();
  predictions[T] = af::max(0,predictions[T]);
}

void ReLUActivation::backprop(){
  gather_delta();
  delta[T] = (predictions[T] > 0) * delta[T];
}

TanhActivation::TanhActivation(unsigned dimensionality) : LinearActivation(dimensionality){}

void TanhActivation::predict(){
  gather_input();
  predictions[T] = af::tanh(predictions[T]); 
}

void TanhActivation::backprop(){
  gather_delta();
  delta[T] =  (1 - af::pow2(predictions[T])) * delta[T];
}
