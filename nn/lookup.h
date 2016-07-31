#ifndef LOOKUP_H
#define LOOKUP_H

#include <cassert>
#include "arrayfire.h"
#include "units.h"
#include "nnutils.h"

using namespace std;

template <class SYMTYPE>
class LinearLookup : public InputSymbolicUnit<SYMTYPE>{
  
public:
 LinearLookup(vector<SYMTYPE> const &dictionary, 
	      unsigned vocab_size, 
	      unsigned embedding_size,
	      SYMTYPE const &unknown_word);//dictionary with unknown word fallback
 LinearLookup(vector<SYMTYPE> const &dictionary, 
	      unsigned vocab_size, 
	      unsigned embedding_size);//dictionary without unknown word fallback
 LinearLookup(vector<SYMTYPE> const &dictionary,
	     af::array &embeddings, 
	     unsigned vocab_size, 
	     bool unk_allowed,
	     bool update_embeddings=false);
             //dictionary from pretrained embeddings with flag indicating if first embedding is used for unknown words

  virtual void predict();
  virtual void backprop();
  virtual void update(float alpha,bool average,bool adagrad);
  virtual void set_input(vector<SYMTYPE> const &xinput);//x symbolic data to be propagated through the network

  virtual af::array& get_predictions() {return xembedded[this->T];};
  virtual unsigned in_dimensionality()const{return vsize*embedding_size;};
  virtual unsigned out_dimensionality()const{return vsize*embedding_size;};
  virtual unsigned vocab_size()const{return vsize;};
  virtual void add_parent(NetworkUnit *parent);      

  virtual void set_seq_length(unsigned L);
  virtual unsigned get_seq_length();

  virtual void swap_weights();

protected:
  void init_stack();

  vector<af::array> xembedded;
  af::array xdictionary;
  af::array jacobian;
  af::array Avgdictionary,Adadictionary;
  float avgN = 0;

  vector<vector<unsigned>> xcodes;
  unsigned vsize; //num discrete symbols on input 
  unsigned embedding_size;  
  BiMap<SYMTYPE> xmapper;
  bool embedding_update = true;//flag telling whether we want to update the embeddings (in case of pretrained embeddings)
private:
  
};

#include "lookup.hpp"
#endif
