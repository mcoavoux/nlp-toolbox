#include "ppa.h"
#include "nnutils.h"
#include <fstream>
#include <set>
#include <cstdlib>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <random>
#include <limits>
#include <omp.h>

PPADataEncoder::PPADataEncoder(const char *filename){add_data(filename);}
PPADataEncoder::PPADataEncoder(vector<string> const &yvalues,vector<vector<string>> const &xvalues){set_data(yvalues,xvalues);}

void PPADataEncoder::add_data(const char *filename){

  ifstream in_file(filename);
  string bfr;
  vector<string> tokens;
  while (getline(in_file,bfr)){
    vector<string> tokens;
    tokenize_dataline(bfr,tokens);
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



float ConditionalDiscreteDistribution::operator()(unsigned valueA){

  assert(K==1);
  assert(valueA < var_dictionaries[0].size());
  unsigned i = 0;
  unsigned j = 0;
  unsigned k = 0;
  unsigned l = valueA ;
  return probs[i][j][k][l];
}

float ConditionalDiscreteDistribution::operator()(unsigned valueA,unsigned cond_valueB){

  assert(K==2);
  assert(valueA < var_dictionaries[0].size());
  assert(cond_valueB < var_dictionaries[1].size());
  unsigned i = 0;
  unsigned j = 0;
  unsigned k = cond_valueB;
  unsigned l = valueA ;
  return probs[i][j][k][l];
}

float ConditionalDiscreteDistribution::operator()(unsigned valueA,unsigned cond_valueB,unsigned cond_valueC){

  assert(K==3);
  assert(valueA < var_dictionaries[0].size());
  assert(cond_valueB < var_dictionaries[1].size());
  assert(cond_valueC < var_dictionaries[2].size());

  unsigned i = 0;
  unsigned j = cond_valueC;
  unsigned k = cond_valueB;
  unsigned l = valueA ;
  return probs[i][j][k][l];
}

float ConditionalDiscreteDistribution::operator()(unsigned valueA,unsigned cond_valueB,unsigned cond_valueC,unsigned cond_valueD){

  assert(K==4);
  assert(valueA < var_dictionaries[0].size());
  assert(cond_valueB < var_dictionaries[1].size());
  assert(cond_valueC < var_dictionaries[2].size());
  assert(cond_valueD < var_dictionaries[3].size());

  unsigned i = cond_valueD;
  unsigned j = cond_valueC;
  unsigned k = cond_valueB;
  unsigned l = valueA ;
  return probs[i][j][k][l];

}



void ConditionalDiscreteDistribution::set_value(string const &valueA,float val){
 
  assert(K==1);
  assert(var_dictionaries[0].has_key(valueA));

  unsigned i = 0;
  unsigned j = 0;
  unsigned k = 0;
  unsigned l = var_dictionaries[0].get_value(valueA);
  probs[i][j][k][l] = val;

}

void ConditionalDiscreteDistribution::set_value(string const &valueA,string const &cond_valueB,float val){
 
  assert(K==2);
  assert(var_dictionaries[0].has_key(valueA));
  assert(var_dictionaries[1].has_key(cond_valueB));
  unsigned i = 0;
  unsigned j = 0;
  unsigned k = var_dictionaries[1].get_value(cond_valueB);
  unsigned l = var_dictionaries[0].get_value(valueA);
  probs[i][j][k][l] = val;
}

void ConditionalDiscreteDistribution::set_value(string const &valueA,string const &cond_valueB,string const &cond_valueC,float val){
 
  assert(K==3);
  assert(var_dictionaries[0].has_key(valueA));
  assert(var_dictionaries[1].has_key(cond_valueB));
  assert(var_dictionaries[2].has_key(cond_valueC));
  unsigned i = 0;
  unsigned j = var_dictionaries[2].get_value(cond_valueC);
  unsigned k = var_dictionaries[1].get_value(cond_valueB);
  unsigned l = var_dictionaries[0].get_value(valueA);
  probs[i][j][k][l] = val;
}

void ConditionalDiscreteDistribution::set_value(string const &valueA,string const &cond_valueB,string const &cond_valueC,string const &cond_valueD,float val){
 
  assert(K==4);
  assert(var_dictionaries[0].has_key(valueA));
  assert(var_dictionaries[1].has_key(cond_valueB));
  assert(var_dictionaries[2].has_key(cond_valueC));
  assert(var_dictionaries[3].has_key(cond_valueD));
  unsigned i = var_dictionaries[3].get_value(cond_valueD);
  unsigned j = var_dictionaries[2].get_value(cond_valueC);
  unsigned k = var_dictionaries[1].get_value(cond_valueB);
  unsigned l = var_dictionaries[0].get_value(valueA);
  probs[i][j][k][l] = val;
}

ConditionalDiscreteDistribution::ConditionalDiscreteDistribution(vector<string> const &domainA){
  var_dictionaries.resize(4);
  var_dictionaries[0] = BiEncoder<string>(domainA,string("___UNK___"));
  K = 1;
  init_matrix();
}

ConditionalDiscreteDistribution::ConditionalDiscreteDistribution(vector<string> const &domainA,vector<string> const &cond_domainB){
  var_dictionaries.resize(4);
  var_dictionaries[0] = BiEncoder<string>(domainA,string("___UNK___"));
  var_dictionaries[1] = BiEncoder<string>(cond_domainB,string("___UNK___"));
  K = 2;
  init_matrix();
}

ConditionalDiscreteDistribution::ConditionalDiscreteDistribution(vector<string> const &domainA,
								 vector<string> const &cond_domainB,
								 vector<string> const &cond_domainC){
  var_dictionaries.resize(4);
  var_dictionaries[0] = BiEncoder<string>(domainA,string("___UNK___"));
  var_dictionaries[1] = BiEncoder<string>(cond_domainB,string("___UNK___"));
  var_dictionaries[2] = BiEncoder<string>(cond_domainC,string("___UNK___"));
  K = 3;
  init_matrix();
}

ConditionalDiscreteDistribution::ConditionalDiscreteDistribution(vector<string> const &domainA,
								 vector<string> const &cond_domainB,
								 vector<string> const &cond_domainC,
								 vector<string> const &cond_domainD){
  var_dictionaries.resize(4);
  var_dictionaries[0] = BiEncoder<string>(domainA,string("___UNK___"));
  var_dictionaries[1] = BiEncoder<string>(cond_domainB,string("___UNK___"));
  var_dictionaries[2] = BiEncoder<string>(cond_domainC,string("___UNK___"));
  var_dictionaries[3] = BiEncoder<string>(cond_domainC,string("___UNK___"));
  K = 4;
  init_matrix();

}

void ConditionalDiscreteDistribution::init_matrix(){
  probs.resize(std::max<unsigned>(var_dictionaries[3].size(),1));
  for(int i = 0; i < probs.size();++i){
    probs[i].resize(std::max<unsigned>(var_dictionaries[2].size(),1));
    for(int j = 0; j < probs[i].size();++j){
      probs[i][j].resize(std::max<unsigned>(var_dictionaries[1].size(),1));
      for(int k = 0; k < probs[i][j].size();++k){
	probs[i][j][k].resize(var_dictionaries[0].size(),std::numeric_limits<float>::min());
      }
    }
  }
}

ostream& ConditionalDiscreteDistribution::display_dimensions(ostream &out){
  out << "0 : " <<  var_dictionaries[0].size() 
      << " 1 : " <<   var_dictionaries[1].size() 
      << " 2 : " << var_dictionaries[2].size() 
      << " 3 : " << var_dictionaries[3].size()<<endl; 
  return out;
}


ConditionalDiscreteDistribution::ConditionalDiscreteDistribution(const char *filename){
  from_file(filename);
}

unsigned ConditionalDiscreteDistribution::get_value_index(string value, unsigned varidx){
  return var_dictionaries[varidx].get_value(value);
}

void ConditionalDiscreteDistribution::get_vardomain(unsigned varidx,BiEncoder<string> &domain)const{
  domain = var_dictionaries[varidx];
}

void ConditionalDiscreteDistribution::set_vardomain(unsigned varidx,BiEncoder<string> const &domain){
  var_dictionaries[varidx] = domain;
}


void ConditionalDiscreteDistribution::from_file(const char *filename,
						vector<string> const &domainA,
						vector<string> const &given_domainB,
						vector<string> const &given_domainC,
						vector<string> const &given_domainD){

  var_dictionaries.resize(4);
  if (domainA.size() > 0){var_dictionaries[0] = BiEncoder<string>(domainA,string("___UNK___"));K=1;}
  if (given_domainB.size() > 0){var_dictionaries[1] = BiEncoder<string>(given_domainB,string("___UNK___"));K=2;}
  if (given_domainC.size() > 0){var_dictionaries[2] = BiEncoder<string>(given_domainC,string("___UNK___"));K=3;}
  if (given_domainD.size() > 0){var_dictionaries[3] = BiEncoder<string>(given_domainD,string("___UNK___"));K=4;}

  init_matrix();

  ifstream infile(filename);
  string bfr;
  while(getline(infile,bfr)){
    vector<string> fields;
    tokenize_dataline(bfr,fields);
    double p = stod(fields.back());
    fields.pop_back();
    switch (fields.size()){
    case 1: 
      assert(K == 1);
      set_value(fields[0],p);
      break;
    case 2:
      assert(K == 2);
      set_value(fields[0],fields[1],p);
      break;
    case 3:
      assert(K == 3);
      set_value(fields[0],fields[1],fields[2],p);
      break;
    case 4:
      assert(K == 4);
      set_value(fields[0],fields[1],fields[2],fields[3],p);
      break;
    }
  }
  infile.close();
}


void ConditionalDiscreteDistribution::from_file(const char *filename){
  
  //first pass = get var domains
  //second pass = fill with probs
  vector<vector<string>> vardomains;
  ifstream infile(filename);
  string bfr;
  while(getline(infile,bfr)){
    vector<string> fields;
    tokenize_dataline(bfr,fields);
    vardomains.push_back(fields);
    vardomains.back().pop_back();
  }
  infile.close();

  //build the matrix
  var_dictionaries.clear();
  var_dictionaries.resize(4);
  for(int i = 0; i < 4;++i){
    if (vardomains[i].size() > 0){
      var_dictionaries[i] = BiEncoder<string>(vardomains[i],string("___UNK___"));
      K = i + 1;
    }else{
      break;
    }
  }
  init_matrix();
  //reread and fill the matrix
  ifstream infile2(filename);
  while(getline(infile2,bfr)){
    vector<string> fields;
    tokenize_dataline(bfr,fields);
    double p = stod(fields.back());
    fields.pop_back();
    switch (fields.size()){
    case 1: 
      assert(K == 1);
      set_value(fields[0],p);
      break;
    case 2:
      assert(K == 2);
      set_value(fields[0],fields[1],p);
      break;
    case 3:
      assert(K == 3);
      set_value(fields[0],fields[1],fields[2],p);
      break;
    case 4:
      assert(K == 4);
      set_value(fields[0],fields[1],fields[2],fields[3],p);
      break;
    }
  }
  infile2.close();
}

float SparseConditionalDiscreteDistribution::operator()(unsigned valueA){ //unconditional... P(X=x)
  unordered_map<quad,float>::iterator gotcha = dict.find(make_tuple(valueA,0,0,0));
  if(gotcha != dict.end()){return gotcha->second;}
  else{return std::numeric_limits<float>::min();}
}

float SparseConditionalDiscreteDistribution::operator()(unsigned valueA,unsigned given_valueB){
  unordered_map<quad,float>::iterator gotcha = dict.find(make_tuple(valueA,given_valueB,0,0));
  if(gotcha != dict.end()){return gotcha->second;}
  else{return std::numeric_limits<float>::min();}
} 

float SparseConditionalDiscreteDistribution::operator()(unsigned valueA,unsigned given_valueB,unsigned given_valueC){ // P(Y=y | X1=x1 X2=x2)
  unordered_map<quad,float>::iterator gotcha = dict.find(make_tuple(valueA,given_valueB,given_valueC,0));
  if(gotcha != dict.end()){return gotcha->second;}
  else{return std::numeric_limits<float>::min();}
}

float SparseConditionalDiscreteDistribution::operator()(unsigned valueA,unsigned given_valueB,unsigned given_valueC,unsigned given_valueD){
  unordered_map<quad,float>::iterator gotcha = dict.find(make_tuple(valueA,given_valueB,given_valueC,given_valueD));
  if(gotcha != dict.end()){return gotcha->second;}
  else{return std::numeric_limits<float>::min();}
}

float SparseConditionalDiscreteDistribution::operator()(string const &valueA){
  unsigned idx = var_dictionaries[0].get_value(valueA);
  return (*this)(idx);
}

float SparseConditionalDiscreteDistribution::operator()(string const &valueA,string const &given_valueB){

  unsigned idx = var_dictionaries[0].get_value(valueA);
  unsigned jdx = var_dictionaries[1].get_value(given_valueB);
  return (*this)(idx,jdx);

}

float SparseConditionalDiscreteDistribution::operator()(string const &valueA,string const &given_valueB,string const &given_valueC){

  unsigned idx = var_dictionaries[0].get_value(valueA);
  unsigned jdx = var_dictionaries[1].get_value(given_valueB);
  unsigned kdx = var_dictionaries[2].get_value(given_valueC);
  return (*this)(idx,jdx,kdx);

}

float SparseConditionalDiscreteDistribution::operator()(string const &valueA,string const &given_valueB,string const &given_valueC,string const &given_valueD){

  unsigned idx = var_dictionaries[0].get_value(valueA);
  unsigned jdx = var_dictionaries[1].get_value(given_valueB);
  unsigned kdx = var_dictionaries[2].get_value(given_valueC);
  unsigned ldx = var_dictionaries[3].get_value(given_valueD);

  return (*this)(idx,jdx,kdx,ldx);

}

void SparseConditionalDiscreteDistribution::set_value(string const &valueA,float val){

  unsigned idx = var_dictionaries[0].get_value(valueA);
  dict[make_tuple(idx,0,0,0)] = val;
}

void SparseConditionalDiscreteDistribution::set_value(string const &valueA,string const &given_valueB,float val){

  unsigned idx = var_dictionaries[0].get_value(valueA);
  unsigned jdx = var_dictionaries[1].get_value(given_valueB);
  dict[make_tuple(idx,jdx,0,0)] = val;

}
void SparseConditionalDiscreteDistribution::set_value(string const &valueA,string const &given_valueB,string const &given_valueC,float val){

  unsigned idx = var_dictionaries[0].get_value(valueA);
  unsigned jdx = var_dictionaries[1].get_value(given_valueB);
  unsigned kdx = var_dictionaries[2].get_value(given_valueC);
  dict[make_tuple(idx,jdx,kdx,0)] = val;
}
void SparseConditionalDiscreteDistribution::set_value(string const &valueA,string const &given_valueB,string const &given_valueC,string const &given_valueD,float val){

  unsigned idx = var_dictionaries[0].get_value(valueA);
  unsigned jdx = var_dictionaries[1].get_value(given_valueB);
  unsigned kdx = var_dictionaries[2].get_value(given_valueC);
  unsigned ldx = var_dictionaries[2].get_value(given_valueD);
  dict[make_tuple(idx,jdx,kdx,ldx)] = val;
}

unsigned SparseConditionalDiscreteDistribution::get_value_index(string value, unsigned varidx){
  return var_dictionaries[varidx].get_value(value);
}

void SparseConditionalDiscreteDistribution::get_vardomain(unsigned varidx,BiEncoder<string> &domain) const{
  domain = var_dictionaries[varidx];
}

void SparseConditionalDiscreteDistribution::set_vardomain(unsigned varidx,BiEncoder<string> const &domain){
  var_dictionaries[varidx] = domain;
}

SparseConditionalDiscreteDistribution::SparseConditionalDiscreteDistribution(const char *filename){
  //from_file(filename);
}


SparseConditionalDiscreteDistribution::SparseConditionalDiscreteDistribution(vector<string> const &domainA){
  var_dictionaries.resize(4);
  var_dictionaries[0] = BiEncoder<string>(domainA,string("___UNK___"));
  K = 1;
}

SparseConditionalDiscreteDistribution::SparseConditionalDiscreteDistribution(vector<string> const &domainA,
									     vector<string> const &given_domainB){
  var_dictionaries.resize(4);
  var_dictionaries[0] = BiEncoder<string>(domainA,string("___UNK___"));
  var_dictionaries[1] = BiEncoder<string>(given_domainB,string("___UNK___"));
  K = 2;
}

SparseConditionalDiscreteDistribution::SparseConditionalDiscreteDistribution(vector<string> const &domainA,
									     vector<string> const &given_domainB,
									     vector<string> const &given_domainC){
  var_dictionaries.resize(4);
  var_dictionaries[0] = BiEncoder<string>(domainA,string("___UNK___"));
  var_dictionaries[1] = BiEncoder<string>(given_domainB,string("___UNK___"));
  var_dictionaries[2] = BiEncoder<string>(given_domainC,string("___UNK___"));
  K = 3;
}

SparseConditionalDiscreteDistribution::SparseConditionalDiscreteDistribution(vector<string> const &domainA,
									     vector<string> const &given_domainB,
									     vector<string> const &given_domainC,
									     vector<string> const &given_domainD){
  var_dictionaries.resize(4);
  var_dictionaries[0] = BiEncoder<string>(domainA,string("___UNK___"));
  var_dictionaries[1] = BiEncoder<string>(given_domainB,string("___UNK___"));
  var_dictionaries[2] = BiEncoder<string>(given_domainC,string("___UNK___"));
  var_dictionaries[3] = BiEncoder<string>(given_domainD,string("___UNK___"));
  K = 4;
}


void SparseConditionalDiscreteDistribution::from_file(const char *filename,
						      vector<string> const &domainA,
						      vector<string> const &given_domainB,
						      vector<string> const &given_domainC,
						      vector<string> const &given_domainD){

  var_dictionaries.resize(4);
  if (domainA.size() > 0){var_dictionaries[0] = BiEncoder<string>(domainA,string("___UNK___"));K=1;}
  if (given_domainB.size() > 0){var_dictionaries[1] = BiEncoder<string>(given_domainB,string("___UNK___"));K=2;}
  if (given_domainC.size() > 0){var_dictionaries[2] = BiEncoder<string>(given_domainC,string("___UNK___"));K=3;}
  if (given_domainD.size() > 0){var_dictionaries[3] = BiEncoder<string>(given_domainD,string("___UNK___"));K=4;}
  ifstream infile(filename);
  string bfr;
  while(getline(infile,bfr)){
    vector<string> fields;
    tokenize_dataline(bfr,fields);
    double p = stod(fields.back());
    fields.pop_back();
    switch (fields.size()){
    case 1: 
      assert(K == 1);
      set_value(fields[0],p);
      break;
    case 2:
      assert(K == 2);
      set_value(fields[0],fields[1],p);
      break;
    case 3:
      assert(K == 3);
      set_value(fields[0],fields[1],fields[2],p);
      break;
    case 4:
      assert(K == 4);
      set_value(fields[0],fields[1],fields[2],fields[3],p);
      break;
    }
  }
  infile.close();
}

DataSampler::DataSampler(const char *original_dataset,
			 const char *vdistrib,
			 const char *x1givenv,
			 const char *pgivenv,
			 const char *x2givenvp,
			 const char *pgivenx1,
			 const char *x2givenx1p){

  this->D = read_dataset(original_dataset);
  vector<string> vdomain;
  vector<string> x1domain;
  vector<string> pdomain;
  vector<string> x2domain;
  make_domains(vdistrib,
	       x1givenv,
	       pgivenv,
	       pgivenx1,
	       x2givenvp,
	       x2givenx1p,
	       vdomain,
	       x1domain,
	       pdomain,
	       x2domain);

  this->vdistrib.from_file(vdistrib,vdomain,vector<string>(),vector<string>(),vector<string>());
  this->x1givenv.from_file(x1givenv,x1domain,vdomain,vector<string>(),vector<string>());
  this->pgivenv.from_file(pgivenv,pdomain,vdomain,vector<string>(),vector<string>());
  this->pgivenx1.from_file(pgivenx1,pdomain,x1domain,vector<string>(),vector<string>());
  this->x2givenx1p.from_file(x2givenx1p,x2domain,x1domain,pdomain,vector<string>());
  this->x2givenvp.from_file(x2givenvp,x2domain,vdomain,pdomain,vector<string>());
  var_dictionaries.push_back(BiEncoder<string>(vdomain,string("___UNK___")));
  var_dictionaries.push_back(BiEncoder<string>(x1domain,string("___UNK___")));
  var_dictionaries.push_back(BiEncoder<string>(pdomain,string("___UNK___")));
  var_dictionaries.push_back(BiEncoder<string>(x2domain,string("___UNK___")));

  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  random_generator =  std::default_random_engine (seed);

}

unsigned DataSampler::read_dataset(const char *original_dataset){//returns the number of datalines

  ifstream infile(original_dataset);
  string bfr;
  vector<string> fields;
  while(getline(infile,bfr)){
    tokenize_dataline(bfr,fields);
    XVALUES.push_back(vector<string>(fields.begin()+1,fields.end()-1));
    YVALUES.push_back(fields.back());
  }
  infile.close();
  return YVALUES.size();
}

void DataSampler::generate_sample(vector<string> &yvalues,vector<vector<string>> &xvalues,unsigned N){
  yvalues.clear();
  xvalues.clear();
  yvalues.resize(N);
  xvalues.resize(N);
  for(int i = 0; i < N;++i){
    string yval;
    xvalues[i] = sample_datum(yval);
    yvalues[i] = yval;
  }
}

void  DataSampler::generate_sample(PPADataEncoder &data_set,unsigned N){

  vector<string> yvals;
  vector<vector<string>> xvals;
  generate_sample(yvals,xvals,N);
  data_set.set_data(yvals,xvals);

}

void DataSampler::getYdictionary(vector<string> &ydict)const{

  ydict.clear();
  set<string> dict;
  for(int i = 0; i < YVALUES.size();++i){
      dict.insert(YVALUES[i]);
  }
  ydict.assign(dict.begin(),dict.end());
}


ostream& DataSampler::dump_sample(ostream &out,vector<string> &yvalues,vector<vector<string>> &xvalues)const{
  
  for(int i = 0; i < yvalues.size();++i){
    for(int j = 0 ; j < xvalues[i].size();++j){
      out << xvalues[i][j] << " ";
    }
    out << yvalues[i] << endl;
  }
  return out;
}


float DataSampler::nominal_prob(unsigned v, unsigned x1, unsigned p,unsigned x2){
  return vdistrib(v) * x1givenv(x1,v) * pgivenx1(p,x1) * x2givenx1p(x2,x1,p);
}

float DataSampler::verbal_prob(unsigned v, unsigned x1, unsigned p,unsigned x2){
  return vdistrib(v) * x1givenv(x1,v) * pgivenv(p,v) * x2givenvp(x2,v,p);
}


void DataSampler::make_domains(const char *vdistrib,
		    const char *x1givenv,
		    const char *pgivenv,
		    const char *pgivenx1,
		    const char *x2givenvp,
		    const char *x2givenx1p,
		    vector<string> &vdomain,
		    vector<string> &x1domain,
		    vector<string> &pdomain,
		    vector<string> &x2domain){

  vdomain.clear();
  x1domain.clear();
  pdomain.clear();
  x2domain.clear();
  
  string bfr;
  vector<string> fields;

  ifstream vstream(vdistrib);
  while(getline(vstream,bfr)){
    tokenize_dataline(bfr,fields);
    vdomain.push_back(fields[0]);
  }
  vstream.close();

  ifstream x1vstream(x1givenv);
  while(getline(x1vstream,bfr)){
    tokenize_dataline(bfr,fields);
    x1domain.push_back(fields[0]);
  }
  x1vstream.close();

  ifstream pstream(pgivenv);
  while(getline(pstream,bfr)){
    tokenize_dataline(bfr,fields);
    pdomain.push_back(fields[0]);
  }
  pstream.close();

  ifstream pstream2(pgivenx1);
  while(getline(pstream2,bfr)){
    tokenize_dataline(bfr,fields);
    pdomain.push_back(fields[0]);
    x1domain.push_back(fields[1]);
  }
  pstream2.close(); 

  
  ifstream x2stream(x2givenx1p);
  while(getline(x2stream,bfr)){
    tokenize_dataline(bfr,fields);
    x2domain.push_back(fields[0]);
  }
  x2stream.close();

  ifstream x2stream2(x2givenvp);
  while(getline(x2stream2,bfr)){
    tokenize_dataline(bfr,fields);
    x2domain.push_back(fields[0]);
  }
  x2stream2.close();
}

vector<string> DataSampler::sample_datum(string &yvalue){
  uniform_int_distribution<int> U(0,(this->D-1)*4);
  int idx = U( random_generator );
  int line_idx = (int)( (float)idx / 4 );
  int col_idx = idx % 4;

  //need to fill domain values
  vector<float> probs(var_dictionaries[col_idx].size(),0.0);

  unsigned v,x1,p,x2;

  v = var_dictionaries[0].get_value(XVALUES[line_idx][0]);
  x1 = var_dictionaries[1].get_value(XVALUES[line_idx][1]);
  p = var_dictionaries[2].get_value(XVALUES[line_idx][2]);
  x2 = var_dictionaries[3].get_value(XVALUES[line_idx][3]);

  if (YVALUES[line_idx] == string("N")){ //attaches to noun
    float Z = 0;
    if (col_idx == 0){
      #pragma omp parallel for reduction(+:Z)
      for(int i = 0; i < var_dictionaries[col_idx].size();++i){    
	probs[i] = nominal_prob(i,x1,p,x2);
	Z += probs[i];
      }
    }
    else if (col_idx == 1){
      #pragma omp parallel for reduction(+:Z)
      for(int i = 0; i < var_dictionaries[col_idx].size();++i){    
	probs[i] = nominal_prob(v,i,p,x2);
	Z += probs[i];
      }
    }
    else if (col_idx == 2){
      #pragma omp parallel for reduction(+:Z)
      for(int i = 0; i < var_dictionaries[col_idx].size();++i){    
	probs[i] = nominal_prob(v,x1,i,x2);
	Z += probs[i];
      }
    }
    else if (col_idx == 3){
      #pragma omp parallel for reduction(+:Z)
      for(int i = 0; i < var_dictionaries[col_idx].size();++i){    
	probs[i] = nominal_prob(v,x1,p,i);
	Z += probs[i];
      }
    }
    #pragma omp parallel for
    for(int i = 0; i < var_dictionaries[col_idx].size();++i){ 
      probs[i] /= Z;
    }
  }

  if (YVALUES[line_idx] == string("V")){ //attaches to verb
    float Z = 0;
    if (col_idx == 0){
      #pragma omp parallel for reduction(+:Z)
      for(int i = 0; i < var_dictionaries[col_idx].size();++i){    
	probs[i] = verbal_prob(i,x1,p,x2);
	Z += probs[i];
      }
    }
    else if (col_idx == 1){
      #pragma omp parallel for reduction(+:Z)
      for(int i = 0; i < var_dictionaries[col_idx].size();++i){    
	probs[i] = verbal_prob(v,i,p,x2);
	Z += probs[i];
      }
    }
    else if (col_idx == 2){
      #pragma omp parallel for reduction(+:Z)
      for(int i = 0; i < var_dictionaries[col_idx].size();++i){    
	probs[i] = verbal_prob(v,x1,i,x2);
	Z += probs[i];
      }
    }
    else if (col_idx == 3){
      #pragma omp parallel for reduction(+:Z)
      for(int i = 0; i < var_dictionaries[col_idx].size();++i){    
	probs[i] = verbal_prob(v,x1,p,i);
	Z += probs[i];
      }
    }
    #pragma omp parallel for
    for(int i = 0; i < var_dictionaries[col_idx].size();++i){ 
      probs[i] /= Z;
    }
  }
  //cumulative distrib
  std::uniform_real_distribution<double> dist(0.0,1.0);
  float r = dist(random_generator);
  float cum_prob = 0;
  for(int i = 0; i < probs.size();++i){
    cum_prob += probs[i];
    if(cum_prob > 1.0 && i != probs.size()-1){cerr << "illegal cum prob : "<< cum_prob << ":" << i << "/"<< probs.size() << endl;}
    if (cum_prob > r){
      vector<string> xsamp(XVALUES[line_idx]);
      xsamp[col_idx] = var_dictionaries[col_idx].get_key(i);
      yvalue = YVALUES[line_idx];
      return xsamp;
    }
  }
  //fall back
  vector<string> xsamp(XVALUES[line_idx]);
  xsamp[col_idx] = var_dictionaries[col_idx].get_key(var_dictionaries[col_idx].size()-1);
  yvalue = YVALUES[line_idx];
  return xsamp;
}


