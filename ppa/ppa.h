#ifndef PPA_H
#define PPA_H

#include <vector>
#include <set>
#include <string>
#include <tuple>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include "arrayfire.h"
#include "utils.h"

using namespace std;


class PPADataEncoder{

  /*
   *  This class encodes the PP attachement data of Ratnaparkhi 1994 into data structures suitable for NN.
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

class ConditionalDiscreteDistribution{
  
 public:
  ConditionalDiscreteDistribution(){this->K=0;};
  ConditionalDiscreteDistribution(unsigned K){this->K=K;};
  float operator()(string const &valueA);//unconditional...
  float operator()(string const &valueA,string const &given_valueB);
  float operator()(string const &valueA,string const &given_valueB,string const &given_valueC);
  float operator()(string const &valueA,string const &given_valueB,string const &given_valueC,string const &given_valueD);
  void set_value(string const &valueA,float val);
  void set_value(string const &valueA,string const &given_valueB,float val);
  void set_value(string const &valueA,string const &given_valueB,string const &given_valueC,float val);
  void set_value(string const &valueA,string const &given_valueB,string const &given_valueC,string const &given_valueD,float val);
  void from_file(const char *filename);
  void get_conditioned_domain(set<string> &set_of_values)const; // returns the set of values of the Y variable from a conditional of the form P(Y|X1... Xn)

protected:

  void set_dimensions(unsigned K){this->K = K;};

private:
  int K; //the number of dimensions (up to 4)
  unordered_map<tuple<string,string,string,string>,float> probs;
};

class DataSampler{
 
public:
  DataSampler(const char *original_dataset,
	      const char *vdistrib,
	      const char *x1givenv,
	      const char *pgivenv,
	      const char *x2givenvp,
	      const char *pgivenx1,
	      const char *x2givenx1p);

  vector<string>& sample_datum(string &yvalue);//no clamping, the method fills yvalue with the actual Y value used
  //vector<string> sample_datum();//clamping

private:
  unsigned read_dataset(const char *original_dataset);//returns the number of datalines
  unsigned D = 0;//number of datalines
  std::default_random_engine random_generator;
  ConditionalDiscreteDistribution vdistrib;
  ConditionalDiscreteDistribution x1givenv;
  ConditionalDiscreteDistribution pgivenv;
  ConditionalDiscreteDistribution pgivenx1;
  ConditionalDiscreteDistribution x2givenvp;
  ConditionalDiscreteDistribution x2givenx1p;
  
  //domains of RV
  vector<vector<string>> domain_values; //v,x1,p,x2 values


  //current data set
  vector<string> YVALUES;
  vector<vector<string>> XVALUES;
};


#endif
