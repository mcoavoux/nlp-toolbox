#ifndef NNUTILS_H
#define NNUTILS_H

#include <unordered_map>
#include <stdexcept>
#include "arrayfire.h"

using namespace std;

class UnknownSymbolException : public std::runtime_error{

public:
  UnknownSymbolException(string const &token):runtime_error((string("(Oops) Symbol not found :") + token)){}
};

/*
 * Random vector/matrix initialization 
 * @param d0: Out Size
 * @param d1: In Size
 * @param n: is the size of input vector
 */
inline void nn_init_weights(af::array &X,unsigned d0,unsigned d1 = 1);       
inline void tokenize_dataline(string const& raw_line,vector<string> &tokens,const char *delim=" \t\n");//line tokenization
inline bool isspace(string &str);

/*
 * Bijective (bidirectional) map that allows to encode decode symbols efficiently back and forth
 */
template<typename SYMTYPE> 
class BiMap{  

public:

  BiMap(){}

  BiMap(vector<SYMTYPE> const &known_values,SYMTYPE const &unk_val);
  BiMap(vector<SYMTYPE> const &known_values);
  BiMap(BiMap const &other);
  BiMap& operator= (BiMap const &other);

  unsigned size()const;

  void insert_value(SYMTYPE const &value);
  
  unsigned operator()(SYMTYPE const &word) const;
  unsigned encode(SYMTYPE const &value)    const;
  void encode_all(vector<SYMTYPE> const &values,
		  vector<unsigned> &codes) const;

  const SYMTYPE& decode(unsigned code)const;

  af::array onehotvector(SYMTYPE const &word);
  af::array onehotmatrix(vector<SYMTYPE> const &wordlist);

private:
  bool unk_word = false;
  vector<SYMTYPE> idx2val;
  unordered_map<SYMTYPE,unsigned> val2idx;
};


template<class OUTSYM,class INSYM>
class DataSampler{
 public:
  void sample_data_set(vector<OUTSYM> &yvalues,vector<vector<INSYM>> &xvalues);
}

#include "nnutils.hpp"
#endif
