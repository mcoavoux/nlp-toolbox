#include <sstream>

/*
 * Performs a random initialisation of a weight vector/matrix.
 * n is the size of the input of the matrix
 */

void nn_init_weights(af::array &X,unsigned d0,unsigned d1){
  float N = d1;
  if (N >= 1){
    X = af::randn(d0,d1) * std::sqrt(2.0/N);
  }else{
    X = af::randn(d0,d1) * 0.01;
  }
}       




/*
void nn_init_weights(af::array &X,unsigned d0,unsigned d1){
  float N = d1;
  X = 2*(af::randu(d0,d1)-0.5);
  af_print(X);
}       
*/

template<class SYMTYPE>
BiMap<SYMTYPE>::BiMap(vector<SYMTYPE> const &known_values,SYMTYPE const &unk_val){
  insert_value(unk_val);
  unk_word = true;
  for(int i = 0; i < known_values.size();++i){
    insert_value(known_values[i]);
  }
}

template<class SYMTYPE>
BiMap<SYMTYPE>::BiMap(vector<SYMTYPE> const &known_values){
  unk_word = false;
  for(int i = 0; i < known_values.size();++i){
    insert_value(known_values[i]);
  }
}
 
template<class SYMTYPE>
BiMap<SYMTYPE>::BiMap(BiMap<SYMTYPE> const &other){
  idx2val = other.idx2val;
  val2idx = other.val2idx;
  unk_word = other.unk_word;
}

template<class SYMTYPE>
unsigned BiMap<SYMTYPE>::size()const{return idx2val.size();}

template<class SYMTYPE>
BiMap<SYMTYPE>& BiMap<SYMTYPE>::operator= (BiMap<SYMTYPE> const &other){
  idx2val = other.idx2val;
  val2idx = other.val2idx;
  unk_word = other.unk_word;
  return *this;
}

template<class SYMTYPE>
void BiMap<SYMTYPE>::insert_value(SYMTYPE const &value){
  typename std::unordered_map<SYMTYPE,unsigned>::const_iterator got = val2idx.find(value);
  if (got == val2idx.end()){
    idx2val.push_back(value);
    val2idx[value] = idx2val.size()-1;
  }
}

template<class SYMTYPE>
unsigned BiMap<SYMTYPE>::operator()(SYMTYPE const &word)const{
  typename std::unordered_map<SYMTYPE,unsigned>::const_iterator got = val2idx.find(word);
  if (got != val2idx.end()){
    return got->second;
  }
  if (unk_word){
    return 0;
  }
  throw UnknownSymbolException(word);
}

template<class SYMTYPE>
af::array BiMap<SYMTYPE>::onehotvector(SYMTYPE const &word){
  af :: array x = af::constant(0,idx2val.size());
  x((*this)(word)) = 1.0;
  return x;
}

template<class SYMTYPE>
af::array BiMap<SYMTYPE>::onehotmatrix(vector<SYMTYPE> const &wordlist){
  af :: array x = af::constant(0,idx2val.size(),wordlist.size());
  for(int i = 0; i < wordlist.size();++i){
    x((*this)(wordlist[i]),i) = 1.0;
  }
  return x;
}


template<class SYMTYPE>
unsigned BiMap<SYMTYPE>::encode(SYMTYPE const &word)const{
  return (*this)(word);
}  

template<class SYMTYPE>
void BiMap<SYMTYPE>::encode_all(vector<SYMTYPE> const &values,vector<unsigned> &codes)const{
  if (codes.size()==values.size()){
    for(int i = 0; i < values.size();++i){
      codes[i] = (*this)(values[i]);
    }
  }else{
    codes.clear();
    for(int i = 0; i < values.size();++i){
      codes.push_back((*this)(values[i]));
    }
  }
}

 
template<class SYMTYPE>
const SYMTYPE& BiMap<SYMTYPE>::decode(unsigned code)const{
  if (code < idx2val.size()){
    return idx2val[code];
  }
  if (unk_word){
    return idx2val[0];
  }
  throw UnknownSymbolException(to_string(code));
} 



