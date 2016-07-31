
RecurrentLinearLayer::RecurrentLinearLayer(unsigned out_size,
					   unsigned in_size){

  this->OutSize = out_size;
  this->InSize  = in_size; 
  nn_init_weights(Weights,OutSize,InSize);
  nn_init_weights(bias,OutSize);
  nn_init_weights(MWeights,OutSize,OutSize);
  nn_init_weights(Mbias,OutSize);
  Jacobian = af::constant(0,OutSize,InSize);
  Jbias    = af::constant(0,OutSize);

  //do a reset method that flushes all the stacks and resets memory
}


void RecurrentLinearLayer::predict(){

  unsigned batch_size = xinput.dims(1);
  ypredictions  = af::matmul(Weights,xmemory) + af::tile(bias,1,batch_size);

}

void RecurrentLinearLayer::backprop(){

  gather_ydelta();
  Jacobian += af::matmulNT(ydelta, xinput);
  Jbias  += af::sum(ydelta, 1);

}

void RecurrentLinearLayer::update(float alpha){
  
  Weights += alpha*Jacobian;
  bias    += alpha*Jbias;

}
void RecurrentLinearLayer::push(){

  xmemory = 

}
void RecurrentLinearLayer::pop();
