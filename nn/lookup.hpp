template <class SYMTYPE>
LinearLookup<SYMTYPE>::LinearLookup(vector<SYMTYPE> const &dictionary, 
				  unsigned vocab_size, 
				  unsigned embedding_size,
				  SYMTYPE const &unknown_word): xmapper(dictionary,unknown_word){

  this->embedding_size = embedding_size;
  this->vsize     = vocab_size;
  nn_init_weights(xdictionary,embedding_size,xmapper.size());
  this->jacobian       = af::constant(0,embedding_size,xmapper.size());
  init_stack();
}


template <class SYMTYPE>
LinearLookup<SYMTYPE>::LinearLookup(vector<SYMTYPE> const &dictionary, 
				  unsigned vocab_size, 
				  unsigned embedding_size): xmapper(dictionary){

  this->embedding_size = embedding_size;
  this->vsize     = vocab_size;
  nn_init_weights(xdictionary,embedding_size,xmapper.size());   //init embeddings randomly
  this->jacobian       = af::constant(0,embedding_size,xmapper.size());
  init_stack();
  
}

template <class SYMTYPE>
LinearLookup<SYMTYPE>::LinearLookup(vector<SYMTYPE> const &dictionary, 
				    af::array &embeddings,
				    unsigned vocab_size, 
				    bool unk_allowed,
                                    bool update_embeddings): xmapper(dictionary,dictionary[0]){

  this->embedding_size = embeddings.dims(0);
  this->vsize     = vocab_size;
  this->xdictionary      = embeddings;
  this->jacobian         = af::constant(0,embedding_size,xmapper.size());
  this->embedding_update = update_embeddings;
  init_stack();
  
}

template <class SYMTYPE>
void LinearLookup<SYMTYPE>::add_parent(NetworkUnit *parent){

  if(!this->parents.empty()){throw InvalidNetworkException();}
  this->parents.push_back(parent);
}     


template <class SYMTYPE>
void LinearLookup<SYMTYPE>::init_stack(){
  set_seq_length(1);
  this->T = 0;  
}

template <class SYMTYPE>
void  LinearLookup<SYMTYPE>::set_seq_length(unsigned L){
  xembedded.resize(L);
  xcodes.resize(L);
}

template <class SYMTYPE>
unsigned  LinearLookup<SYMTYPE>::get_seq_length(){
  return xembedded.size();
}


template <class SYMTYPE>
void LinearLookup<SYMTYPE>::set_input(vector<SYMTYPE> const &xinput){//x symbolic data to be propagated through the network
  xmapper.encode_all(xinput,xcodes[this->T]);  
}


template <class SYMTYPE>
void LinearLookup<SYMTYPE>::predict(){

   assert(xcodes[this->T].size() % vsize == 0);
   unsigned batchSize = xcodes[this->T].size()/vsize;
   af::array vocab_idxes(xcodes[this->T].size(),xcodes[this->T].data());
   xembedded[this->T] = af::moddims(xdictionary(af::span,vocab_idxes),vsize*embedding_size,batchSize);
   //af_print(xembedded[this->T]);
   //cout << "xT="<<this->T<<endl;
}


template <class SYMTYPE>
void LinearLookup<SYMTYPE>::backprop(){
  if(embedding_update){
    af::array M = af::moddims(this->parents[0]->get_delta(),embedding_size,xcodes[this->T].size()); 
    for(int i = 0 ; i < xcodes[this->T].size();++i){
      jacobian(af::span,xcodes[this->T][i]) += M(af::span,i);//sparse matrix style op.
    }
  }
}

template <class SYMTYPE>
void LinearLookup<SYMTYPE>::update(float alpha,bool average,bool adagrad){
  
  if(embedding_update){
    if (adagrad){                          //ADAGRAD AND LEARNING RATE

      if (Adadictionary.isempty()){
	Adadictionary = af::pow2(af::constant(std::numeric_limits<float>::min(),xdictionary.dims(0),xdictionary.dims(1)));
      }
      xdictionary   +=  alpha * (jacobian / af::sqrt(Adadictionary));
      Adadictionary += af::pow2(jacobian);

    }else{  //no adagrad apply regular learning rate
      xdictionary += alpha*jacobian;
    }
 
    if (average){                            //MOVING AVERAGE
      if (Avgdictionary.isempty()){
	Avgdictionary = af::constant(0,xdictionary.dims(0),xdictionary.dims(1));
	avgN = 0;
      }
      //as cumulative moving average
      Avgdictionary = Avgdictionary + (xdictionary - Avgdictionary)/(avgN+1);
      ++avgN;
    }
    jacobian = af::constant(0,embedding_size,xmapper.size());
  }
}


template <class SYMTYPE>
void LinearLookup<SYMTYPE>::swap_weights(){

  if (avgN > 0){
    af::array tmpdict = xdictionary;
    xdictionary       = Avgdictionary;
    Avgdictionary     = tmpdict;
  }
}
