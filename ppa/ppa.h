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
  unsigned get_vocabulary_size()const{return keys.size();};
  
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


class SparseConditionalDiscreteDistribution{
public:
  SparseConditionalDiscreteDistribution(){};
  SparseConditionalDiscreteDistribution(const char *filename);
  SparseConditionalDiscreteDistribution(vector<string> const &domainA);
  SparseConditionalDiscreteDistribution(vector<string> const &domainA,vector<string> const &given_domainB);
  SparseConditionalDiscreteDistribution(vector<string> const &domainA,vector<string> const &given_domainB,vector<string> const &given_domainC);
  SparseConditionalDiscreteDistribution(vector<string> const &domainA,vector<string> const &given_domainB,vector<string> const &given_domainC,vector<string> const &given_domainD);

  unsigned get_value_index(string value, unsigned varidx);

  float operator()(unsigned valueA);//unconditional... P(X=x)
  float operator()(unsigned valueA,unsigned given_valueB); // P(Y=y | X=x)
  float operator()(unsigned valueA,unsigned given_valueB,unsigned given_valueC); // P(Y=y | X1=x1 X2=x2)
  float operator()(unsigned valueA,unsigned given_valueB,unsigned given_valueC,unsigned given_valueD);
  float operator()(string const &valueA);//unconditional...
  float operator()(string const &valueA,string const &given_valueB);
  float operator()(string const &valueA,string const &given_valueB,string const &given_valueC);
  float operator()(string const &valueA,string const &given_valueB,string const &given_valueC,string const &given_valueD);

  void get_vardomain(unsigned varidx,BiEncoder<string> &domain) const;
  void set_vardomain(unsigned varidx,BiEncoder<string> const &domain);

  void from_file(const char *filename,vector<string> const &domainA,vector<string> const &given_domainB,vector<string> const &given_domainC,vector<string> const &given_domainD);
 protected:
  void set_value(string const &valueA,float val);
  void set_value(string const &valueA,string const &given_valueB,float val);
  void set_value(string const &valueA,string const &given_valueB,string const &given_valueC,float val);
  void set_value(string const &valueA,string const &given_valueB,string const &given_valueC,string const &given_valueD,float val);

 private:
  int K;
  typedef tuple<unsigned,unsigned,unsigned,unsigned> quad;
  std::unordered_map<quad,float> dict;
  vector<BiEncoder<string>> var_dictionaries;
  
};

class ConditionalDiscreteDistribution{
  
 public:
  ConditionalDiscreteDistribution(){};
  ConditionalDiscreteDistribution(const char *filename);
  ConditionalDiscreteDistribution(vector<string> const &domainA);
  ConditionalDiscreteDistribution(vector<string> const &domainA,vector<string> const &given_domainB);
  ConditionalDiscreteDistribution(vector<string> const &domainA,vector<string> const &given_domainB,vector<string> const &given_domainC);
  ConditionalDiscreteDistribution(vector<string> const &domainA,vector<string> const &given_domainB,vector<string> const &given_domainC,vector<string> const &given_domainD);

  unsigned get_value_index(string value, unsigned varidx);


  float operator()(unsigned valueA);//unconditional... P(X=x)
  float operator()(unsigned valueA,unsigned given_valueB); // P(Y=y | X=x)
  float operator()(unsigned valueA,unsigned given_valueB,unsigned given_valueC); // P(Y=y | X1=x1 X2=x2)
  float operator()(unsigned valueA,unsigned given_valueB,unsigned given_valueC,unsigned given_valueD);

  float operator()(string const &valueA);//unconditional...
  float operator()(string const &valueA,string const &given_valueB);
  float operator()(string const &valueA,string const &given_valueB,string const &given_valueC);
  float operator()(string const &valueA,string const &given_valueB,string const &given_valueC,string const &given_valueD);

  void get_vardomain(unsigned varidx,BiEncoder<string> &domain) const;
  void set_vardomain(unsigned varidx,BiEncoder<string> const &domain);

  void from_file(const char *filename);
  void from_file(const char *filename,vector<string> const &domainA,vector<string> const &given_domainB,vector<string> const &given_domainC,vector<string> const &given_domainD);

  ostream& display_dimensions(ostream &out);

protected:

  void set_dimensions(unsigned K){this->K = K;};
  void init_matrix();
  void set_value(string const &valueA,float val);
  void set_value(string const &valueA,string const &given_valueB,float val);
  void set_value(string const &valueA,string const &given_valueB,string const &given_valueC,float val);
  void set_value(string const &valueA,string const &given_valueB,string const &given_valueC,string const &given_valueD,float val);

private:
  int K; //the number of dimensions (up to 4)
  vector<vector<vector<vector<float>>>> probs;
  vector<BiEncoder<string>> var_dictionaries;
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

  void getYdictionary(vector<string> &ydict)const;

  vector<string> sample_datum(string &yvalue);
  void generate_sample(vector<string> &yvalues,vector<vector<string>> &xvalues,unsigned N);
  void generate_sample(PPADataEncoder &data_set,unsigned N);
  ostream& dump_sample(ostream &out,vector<string> &yvalues,vector<vector<string>> &xvalues)const;

  //default sampling size = original data size
  unsigned default_size()const{return YVALUES.size();}

  void make_domains(const char *vdistrib,
		    const char *x1givenv,
		    const char *pgivenv,
		    const char *pgivenx1,
		    const char *x2givenvp,
		    const char *x2givenx1p,
		    vector<string> &vdomain,
		    vector<string> &x1domain,
		    vector<string> &pdomain,
		    vector<string> &x2domain);

protected:
  float nominal_prob(unsigned v, unsigned x1, unsigned p,unsigned x2);
  float verbal_prob(unsigned v, unsigned x1, unsigned p,unsigned x2);

private:
  unsigned read_dataset(const char *original_dataset);//returns the number of datalines
  unsigned D = 0;//number of datalines
  std::default_random_engine random_generator;
  ConditionalDiscreteDistribution vdistrib;
  ConditionalDiscreteDistribution x1givenv;
  ConditionalDiscreteDistribution pgivenv;
  ConditionalDiscreteDistribution pgivenx1;
  SparseConditionalDiscreteDistribution x2givenvp;
  SparseConditionalDiscreteDistribution x2givenx1p;
 
  //domains of RV
  vector<BiEncoder<string>> var_dictionaries;//v,x1,p,x2 domains


  //current data set
  vector<string> YVALUES;
  vector<vector<string>> XVALUES;
};


#endif
