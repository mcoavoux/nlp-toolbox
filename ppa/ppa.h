#ifndef PPA_H
#define PPA_H

#include <vector>
#include <string>
#include <tuple>
#include <unordered_map>
#include "arrayfire.h"
#include "utils.h"

using namespace std;


class PPADataEncoder{
  /*
   * This class encodes the PP attachement data of Ratnaparkhi 1994 into data structures suitable for NN.
   */
public:

  PPADataEncoder(const char *filename);
  PPADataEncoder(vector<string> const &yvalues,vector<vector<string>> const &xvalues);
  
  void add_data(const char *filename);
  void set_data(vector<string> const &yvalues,vector<vector<string>> const &xvalues);

  void clear();

  void getYdictionary(vector<string> &ydict)const;
  void getXdictionary(vector<string> &xdict)const;
  void getYdata(vector<string> &ydata)const;
  void getXdata(vector<string> &xdata)const;

  unsigned x_vocab_size()const{return 4;};//number of x symbols on each data line (they share the same embeddings)

private:
  vector<string> yvalues;
  vector<string> xvalues;


};

class Word2vec{

public:

  Word2vec(){};  //word2vec filereader
  Word2vec(const char *filename){};  //word2vec filereader

  void load_dictionary(const char *filename);

  vector<string>& get_keys(){return keys;};
  af::array& get_values(){return dict;};
  unsigned   get_embeddings_dimensions()const{return K;};

  void filter(vector<string> const &word_set);//the param must be a *set* of words that we intersect with the dict

protected:
  void get_dimensions(const char *filename,unsigned &N,unsigned &K);
  void read_line(unsigned idx,vector<float> &xvalues);

private:
  vector<string> keys;
  af::array dict;
  unsigned K;
};



class QuadSampler : public DataSampler{

public:

  ~QuadSampler();
  QuadSampler(const char *distrib_filename,const char *dataset_filename);

  void original_data_set(vector<string> &yvalues, vector<vector<string>> &xvalues);
  void sample_data_set(vector<string> &yvalues,vector<vector<string>> &xvalues);
  void sample_example(vector<string> &yvalues,vector<vector<string>> &xvalues,unsigned idx);//samples only example idx
  void getXdictionary(vector<string> &dictionary);

protected:
  void read_cond_distributions(const char *distrib_filename);
  void read_dataset(const char *dataset_filename);

private:
  unordered_map<tuple<string,string,string>,CountDictionary<string>*> cond_distribs; //head / head-cat / prep 
  vector<string> yvalues;
  vector<vector<string>> xvalues;
};




#endif
