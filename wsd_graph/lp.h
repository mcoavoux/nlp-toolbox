#include "arrayfire.h"
#include "utils.h"

class WSDGraph;

//I/O basic stuff
inline void tokenize_dataline(wstring const& raw_line,vector<wstring> &tokens,const wchar_t *delim=L" \t\n");//line tokenization
inline bool isspace(wstring &str);


class Proximity{ //Default is a gaussian ball style proximity 
  //computes exp(-sqeuclidean)

public:
  Proximity(){dist=0;}
  void aggregate(vector<float> const &xvec,vector<float> const &yvec,float weight=1.0); //computes the proximity between two vectors
  float proximity();//applies global mods such as sqrt, turns dist in prox etc...

protected:
  float dist;
};

class Word2vec{

  //expected input format: word_cat (float)+
public:
  Word2vec(){}
  Word2vec(const char *filename);
  Word2vec(Word2vec const &other);
  Word2vec& operator=(Word2vec const &other);


  void get_value(wstring const &wordform,vector<float> &value);
  void read_dictionary(const char *filename);

  bool empty()const;
  void clear();
  void clear_stats();
  float get_prop_lookup_success()const;

protected:
  void make_dictionary(const char *filename); //fills the main dictionary
  void make_aggregates(); //fills the backoff dictionary

 private:
  Dictionary<wstring,vector<float>> word_embeddings; //keys = <word_category> ;
  Dictionary<wstring,vector<float>> cat_embeddings; //keys = categories ; contains averaged embeddings for fallback on unk words
  unsigned lookup_success, lookupN;
};



//representation
class WSDdataset{
  /**
   * Reads in a tab separated data set with class label in first column.
   * A class label with '?' is an unknown class label
   * Other columns are X values.
   * Vectorization is by default one hot. May use a word embedding dict otherwise for embedding_idxes columns
   */
public:
  

  WSDdataset();
  WSDdataset(char const *datafilename);

  WSDdataset(char const *datafilename,
	     char const *embeddings_filename,
	     float regular_weights=1.0,
	     float embedding_weights=1.0);

  void load_data(char const *datafilename);
  void load_embeddings(char const *embeddings_filename,float regular_weights=1.0,float embedding_weights=1.0);
  void get_graph(WSDGraph &graph,Proximity *distfunc);

  wostream& dump_Ycodes(wostream &out);//dumps the encoding of the Y classes
    
protected:
  void read_dataset(char const *datafilename);
  void code_dataline(vector<wstring> const &line);
  void code_dataset(char const *datafilename, vector<unsigned> const &dense_idxes);

private: 
  vector<wstring> ydata; 
  vector<vector<wstring>> xdata;
  Dictionary<wstring,vector<float>> yencodings;
  vector<Dictionary<wstring,vector<float>>> xencodings; //one dictionary per column
  Word2vec embeddings;
  vector<bool> embedding_codes;//tells which X chunk has to be coded by embeddings
  float reg_weight,ebd_weight;
};


//algorithms
class WSDGraph{
  
public:

  WSDGraph(){}
  WSDGraph(vector<vector<float>> const &xmatrix,vector<vector<float>> const &ylabel,unsigned last_label_idx);//last labelled line idx

  void set_transitions(vector<vector<float>> const &xmatrix);
  void set_labels(vector<vector<float>> const &ylabel,unsigned last_label_idx);//last labelled line idx

  //graph structuration
  void nullify_diag(float zerovalue = 0);//sets the reflexive arc weights to 0

  //note : knn nullifies the reflexive arcs
  void knn(unsigned k);//computes a binary matrix with structural zeros, uses it as a mask to update the transition matrix
  void eps_nn(float eps);//computes a binary matrix with structural zeros, uses it as a mask to update the transition matrix

  void label_propagation(float convergence_eps=0.1);
  //...
  
  wostream& json(wostream &out);//networkx interpretable JSON

protected:
  void row_normalize_matrix(af::array &mat);


private:
  af::array X;//weighted graph edges
  af::array Y;//Y distributions
  unsigned LN; //N Labelled examples
};


