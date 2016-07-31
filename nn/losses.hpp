
af::array soft_divide(const af::array &a, const af::array &b){
    return a / b;
}

template<class SYMTYPE>
SoftMaxLoss<SYMTYPE>::SoftMaxLoss(vector<SYMTYPE> const &dict): ymapper(dict)
{
  Size=ymapper.size(); 
  init_stack();
}


template<class SYMTYPE>
void SoftMaxLoss<SYMTYPE>::init_stack(){

  yreference.push_back(af::constant(0,Size));
  ypredictions.push_back(af::constant(0,Size));
  xdelta.push_back(af::constant(0,Size));
  xinput.push_back(af::constant(0,Size));
  this->T = 0;
}

template<class SYMTYPE>
void SoftMaxLoss<SYMTYPE>::set_seq_length(unsigned L){
  yreference.resize(L);
  ypredictions.resize(L);
  xdelta.resize(L);
  xinput.resize(L);
}

template<class SYMTYPE>
unsigned SoftMaxLoss<SYMTYPE>::get_seq_length(){
  return ypredictions.size();
}



template<class SYMTYPE>
void SoftMaxLoss<SYMTYPE>::gather_xinput(){

  xinput[this->T] =  this->children[0]->get_predictions();
  for(int i = 1; i < this->children.size();++i){
    xinput[this->T] += this->children[i]->get_predictions();
  }

}

template<class SYMTYPE>
void SoftMaxLoss<SYMTYPE>::predict(){
   
  gather_xinput();
  ypredictions[this->T] = af::exp(xinput[this->T]-tile(max(xinput[this->T],0),xinput[this->T].dims(0)));//slower but avoids overflows
  af::array Z =  af::sum(ypredictions[this->T], 0);
  ypredictions[this->T] = af::batchFunc(ypredictions[this->T], Z, soft_divide);  
  //cout << "yT=" << this->T<<endl;
}

template<class SYMTYPE>
void SoftMaxLoss<SYMTYPE>::backprop(){
  xdelta[this->T] = yreference[this->T] - ypredictions[this->T];
  
  //af_print(yreference[this->T]);
  //af_print(ypredictions[this->T]);
  //cout <<"T="<< this->T << endl;
 
}

template<class SYMTYPE>
float SoftMaxLoss<SYMTYPE>::get_loss()const{
  return -af::sum<float>(af::log(ypredictions[this->T]) * yreference[this->T]);
}

template<class SYMTYPE>
vector<SYMTYPE>& SoftMaxLoss<SYMTYPE>::extract_argmaxes(){
  
  af::array m,argmax;
  af::max(m,argmax,ypredictions[this->T],0);
  unsigned *host_argmax = argmax.host<unsigned>();
  unsigned N = argmax.dims(1);

  if (argmaxes.size() != N){argmaxes.resize(N);}
  for(int i = 0; i < N ;++i){argmaxes[i] = ymapper.decode(host_argmax[i]);}
  delete[] host_argmax;
  return argmaxes;
}

template<class SYMTYPE>
void SoftMaxLoss<SYMTYPE>::set_reference(vector<SYMTYPE> const &yreference){
  this->yreference[this->T] = ymapper.onehotmatrix(yreference);
}
