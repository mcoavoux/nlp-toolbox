#include "ppa.h"
#include "nnutils.h"
#include <fstream>
#include <set>
#include <cstdlib>
#include <algorithm>

PPADataEncoder::PPADataEncoder(const char *filename){add_data(filename);}
PPADataEncoder::PPADataEncoder(vector<string> const &yvalues,vector<vector<string>> const &xvalues){set_data(yvalues,xvalues);}

void PPADataEncoder::add_data(const char *filename){

  ifstream in_file(filename);

  string bfr;
  vector<string> tokens;
  while (getline(in_file,bfr)){
    vector<string> tokens;
    tokenize_dataline(bfr,tokens," ");
    yvalues.push_back(tokens[5]);
    xvalues.push_back(tokens[1]); // verb form
    xvalues.push_back(tokens[2]); // noun form
    xvalues.push_back(tokens[3]); // prep form
    xvalues.push_back(tokens[4]); // noun form
  }
  in_file.close();
}
void PPADataEncoder::set_data(vector<string> const &yvalues,vector<vector<string>> const &xvalues){
  
  this->yvalues = yvalues;
  this->xvalues.clear();
  for(int i = 0; i < xvalues.size();++i){
    this->xvalues.push_back(xvalues[i][0]); // verb form
    this->xvalues.push_back(xvalues[i][1]); // noun form
    this->xvalues.push_back(xvalues[i][2]); // prep form
    this->xvalues.push_back(xvalues[i][3]); // noun form
  }
}

void PPADataEncoder::clear(){

  yvalues.clear();
  xvalues.clear();  

}

void PPADataEncoder::getYdata(vector<string> &ydata)const{ydata = this->yvalues;};
void PPADataEncoder::getXdata(vector<string> &xdata)const{xdata = this->xvalues;};

void PPADataEncoder::getYdictionary(vector<string> &ydict)const{

    ydict.clear();
    std::set<string> yset(yvalues.begin(),yvalues.end());
    ydict.assign(yset.begin(),yset.end());

}

void PPADataEncoder::getXdictionary(vector<string> &xdict)const{

    xdict.clear();
    std::set<string> xset(xvalues.begin(),xvalues.end());
    xdict.assign(xset.begin(),xset.end());

}

void Word2vec::filter(vector<string> const &word_set){
 
  set<string> tmp_set(word_set.begin(),word_set.end());
  vector<string> tmp_keys;
  af::array tmp_values = af::constant(0,K,tmp_set.size());
  int j = 0;
  for(int i = 0; i < keys.size();++i){
    if(tmp_set.find(keys[i]) != tmp_set.end()){
      tmp_keys.push_back(keys[i]);
      tmp_values(af::span,j) = dict(af::span,i);
      ++j;
    }
  }
  dict = tmp_values(af::span,af::seq(tmp_keys.size()));
  keys = tmp_keys;
}


void Word2vec::load_dictionary(const char *filename){

   cerr << "loading embeddings..." << endl;
   ifstream in_file(filename);
   unsigned N;
   get_dimensions(filename,N,K);
   dict = af::constant(0,K,N);
   keys.clear();

   vector<string> xvalues;
   string bfr;
   unsigned idx = 0;   
   while (std::getline(in_file, bfr)){
    tokenize_dataline(bfr,xvalues," ");
    keys.push_back(xvalues[0]);
    vector<float> vals(K);
    for(int i = 1; i < xvalues.size();++i){
       vals[i] = std::strtof(xvalues[i].c_str(),NULL);
    }
    af::array gpu_bfr(K,vals.data());
    dict(af::span,idx) = gpu_bfr;
    ++idx;
  }
   cerr << "embeddings loaded..." << endl;
}

void Word2vec::get_dimensions(const char *filename,unsigned &N,unsigned &K){

  ifstream in_file(filename);
  std::string unused;

  getline(in_file, unused);
  vector<string> xvalues;
  tokenize_dataline(unused,xvalues," ");
  K = xvalues.size()-1;
  N = 1;
  while ( std::getline(in_file, unused) ){++N;}
  in_file.close();

}

/*
QuadSampler::QuadSampler(const char *distrib_filename,const char *dataset_filename){

  read_cond_distributions(distrib_filename);
  read_dataset(dataset_filename);

}

QuadSampler::~QuadSampler(){
    for(unordered_map<tuple<string,string,string>,CountDictionary<string>*>::iterator it = cond_distribs.begin();it != cond_distribs.end();++it){
      delete it->second;
    }
}


void QuadSampler::original_data_set(vector<string> &yvalues, vector<vector<string>> &xvalues){
  
  yvalues = this->yvalues;
  xvalues = this->xvalues;
  
}

void QuadSampler::sample_data_set(vector<string> &yvalues,vector<vector<string>> &xvalues){

  yvalues = this->yvalues;
  xvalues = this->xvalues;
  typedef unordered_map<tuple<string,string,string>,CountDictionary<string>*>::iterator ITR;
  string verb("V"); 
  for(int i = 0; i < this->xvalues.size();++i){
    if(yvalues[i] == verb){//sample another verbal dependant
      ITR CD = cond_distribs.find(make_tuple(xvalues[i][0],string("verb"),xvalues[i][2]));
      if(CD != cond_distribs.end()){
	xvalues[i][3] = CD->second->sample();
      }
    }else{
      ITR CD = cond_distribs.find(make_tuple(xvalues[i][0],string("noun"),xvalues[i][2]));
      if(CD != cond_distribs.end()){
	xvalues[i][3] = CD->second->sample();
      }
    }
  }
}


void QuadSampler::sample_example(vector<string> &yvalues,vector<vector<string>> &xvalues,unsigned idx){//samples only example idx

  typedef unordered_map<tuple<string,string,string>,CountDictionary<string>*>::iterator ITR;

  string verb("V"); 
  if(yvalues[idx] == verb){//sample another verbal dependant
      ITR CD = cond_distribs.find(make_tuple(xvalues[idx][0],string("verb"),xvalues[idx][2]));
      if(CD != cond_distribs.end()){
	xvalues[idx][3] = CD->second->sample();
      }
  }else{
      ITR CD = cond_distribs.find(make_tuple(xvalues[idx][0],string("noun"),xvalues[idx][2]));
      if(CD != cond_distribs.end()){
	xvalues[idx][3] = CD->second->sample();
      }
  }
  for(int i = 0; i < xvalues[idx].size();++i){
    cout << xvalues[idx][i] << " ";
  }
  cout << yvalues[idx] << endl;
}


void QuadSampler::read_cond_distributions(const char *distrib_filename){
  
  ifstream in_file(distrib_filename);
  // unordered_map<tuple<string,string,string>,CountDictionary<string>*> cond_distribs; //head / head-cat / prep
  string bfr;
  string prev_head,prev_prep;
  CountDictionary<string> *current_dict;
  while(getline(in_file,bfr)){
    if(bfr.empty()){continue;}
    vector<string> tokens;
    tokenize_dataline(bfr,tokens);

    if(tokens[2] == string("obj")){continue;}//skip additional prep='obj' line returned by sketch-engine

    if (tokens[0] != prev_head || tokens[2] != prev_prep){

      std::tuple<string,string,string> key = std::make_tuple(tokens[0],tokens[1],tokens[2]);
      if (cond_distribs.find(key) == cond_distribs.end()){
	current_dict = new CountDictionary<string>(0);
	cond_distribs[key] = current_dict;
      }else{
	current_dict = cond_distribs[key];
      }
      prev_head = tokens[0];
      prev_prep = tokens[2];
    }
    current_dict->add_count(tokens[3],std::stof(tokens[5]));

  }
  in_file.close();
}


void QuadSampler::read_dataset(const char *dataset_filename){

  ifstream in_file(dataset_filename);
  string bfr;
  while(getline(in_file,bfr)){
    vector<string> tokens;
    tokenize_dataline(bfr,tokens);
    yvalues.push_back(tokens.back());
    xvalues.push_back(vector<string>(tokens.begin()+1,tokens.end()-1));
  }
  in_file.close();
}


void QuadSampler::getXdictionary(vector<string> &dictionary){

  unordered_set<string> dict;
  vector<vector<string>> xvalues;
  //original data set...
  for(int k = 0; k < xvalues.size();++k){dict.insert(xvalues[k].begin(),xvalues[k].end());}
  //sampler vocabulary
  unordered_map<tuple<string,string,string>,CountDictionary<string>*> cond_distribs;
  for(unordered_map<tuple<string,string,string>,CountDictionary<string>*>::iterator it = cond_distribs.begin();it!=cond_distribs.end();++it){
    CountDictionary<string> *c = it->second;
    vector<string> keys;
    c->getXdictionary(keys);
    dict.insert(keys.begin(),keys.end());
  }
  dictionary.clear();
  for(unordered_set<string>::iterator it = dict.begin();it!=dict.end();++it){
    dictionary.push_back(*it);
  }
}
*/
