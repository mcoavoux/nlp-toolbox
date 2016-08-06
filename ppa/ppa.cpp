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

float ConditionalDiscreteDistribution::operator()(string const &cond_valueA){

  assert(K==1);
  tuple<string,string,string,string> key = make_tuple(cond_valueA,string(),string(),string());
  if (probs.find(key) == probs.end()){return std::numeric_limits<float>::epsilon();}
  return probs[key];

}

float ConditionalDiscreteDistribution::operator()(string const &cond_valueA,string const &cond_valueB){

  assert(K==2);
  tuple<string,string,string,string> key = make_tuple(cond_valueA,cond_valueB,string(),string());
  if (probs.find(key) == probs.end()){return std::numeric_limits<float>::epsilon();}
  return probs[key];

}

float ConditionalDiscreteDistribution::operator()(string const &cond_valueA,string const &cond_valueB,string const &cond_valueC){

  assert(K==3);
  tuple<string,string,string,string> key = make_tuple(cond_valueA,cond_valueB,cond_valueC,string());
  if (probs.find(key) == probs.end()){return std::numeric_limits<float>::epsilon();}
  return probs[key];


}

float ConditionalDiscreteDistribution::operator()(string const &cond_valueA,string const &cond_valueB,string const &cond_valueC,string const &cond_valueD){

  assert(K==4);
  tuple<string,string,string,string> key = make_tuple(cond_valueA,cond_valueB,cond_valueC,cond_valueD);
  if (probs.find(key) == probs.end()){return std::numeric_limits<float>::epsilon();}
  return probs[key];

}

void ConditionalDiscreteDistribution::set_value(string const &cond_valueA,float val){
 
  assert(K==1);
  tuple<string,string,string,string> key = make_tuple(cond_valueA,string(),string(),string());
  probs[key] = val;

}
void ConditionalDiscreteDistribution::set_value(string const &cond_valueA,string const &cond_valueB,float val){

  assert(K==2);
  tuple<string,string,string,string> key = make_tuple(cond_valueA,cond_valueB,string(),string());
  probs[key] = val;

}

void ConditionalDiscreteDistribution::set_value(string const &cond_valueA,string const &cond_valueB,string const &cond_valueC,float val){
  
  assert(K==3);
  tuple<string,string,string,string> key = make_tuple(cond_valueA,cond_valueB,cond_valueC,string());
  probs[key] = val;

}

void ConditionalDiscreteDistribution::set_value(string const &cond_valueA,string const &cond_valueB,string const &cond_valueC,string const &cond_valueD,float val){
  
  assert(K==4);
  tuple<string,string,string,string> key = make_tuple(cond_valueA,cond_valueB,cond_valueC,cond_valueD);
  probs[key] = val;

}

// returns the set of values of the Y variable from a conditional of the form P(Y|X1... Xn)
void ConditionalDiscreteDistribution::get_conditioned_domain(set<string> &values)const{ 

  for (unordered_map<tuple<string,string,string,string>,float>::const_iterator it = probs.begin(); it != probs.end();++it){
    values.insert(get<0>(it->first));
  }
}

void ConditionalDiscreteDistribution::from_file(const char *filename){


  ifstream infile(filename);
  string bfr;
  getline(infile,bfr);
  vector<string> fields;
  tokenize_dataline(bfr,fields);
  double p = stod(fields.back());
  switch (fields.size()-1){
  case 1:
    this->K = 1;
    set_value(fields[0],p);
    break;
  case 2:
    this->K = 2;
    set_value(fields[0],fields[1],p);
    break;
  case 3:
    this->K = 3;
    set_value(fields[0],fields[1],fields[2],p);
    break;
  case 4:
    this->K = 4;
    set_value(fields[0],fields[1],fields[2],fields[3],p);
    break;
  default:
    cerr << "conditional distrib dimensionality error\naborting."<<endl;
    exit(1);
  }
  while(getline(infile,bfr)){
    tokenize_dataline(bfr,fields);
    double p = stod(fields.back());
    assert(fields.size()-1 == K);
    switch (fields.size()-1){
    case 1:
      set_value(fields[0],p);
      break;
    case 2:
      set_value(fields[0],fields[1],p);
      break;
    case 3:
      set_value(fields[0],fields[1],fields[2],p);
      break;
    case 4:
      set_value(fields[0],fields[1],fields[2],fields[3],p);
      break;
    default:
      cerr << "conditional distrib dimensionality error\naborting."<<endl;
      exit(1);
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

  set<string> v_values;
  set<string> x1_values;
  set<string> p_values;
  set<string> x2_values;
  this->D = read_dataset(original_dataset);
  this->vdistrib.from_file(vdistrib);
  this->vdistrib.get_conditioned_domain(v_values);
  this->x1givenv.from_file(x1givenv);
  this->x1givenv.get_conditioned_domain(x1_values);
  this->pgivenv.from_file(pgivenv);
  this->pgivenv.get_conditioned_domain(p_values);
  this->pgivenx1.from_file(pgivenx1);
  this->pgivenx1.get_conditioned_domain(p_values);
  this->x2givenvp.from_file(x2givenvp);
  this->x2givenvp.get_conditioned_domain(x2_values);
  this->x2givenx1p.from_file(x2givenx1p);
  this->x2givenx1p.get_conditioned_domain(x2_values);

  domain_values.push_back(vector<string>(v_values.begin(),v_values.end()));
  domain_values.push_back(vector<string>(x1_values.begin(),x1_values.end()));
  domain_values.push_back(vector<string>(p_values.begin(),p_values.end()));
  domain_values.push_back(vector<string>(x2_values.begin(),x2_values.end()));

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


vector<string>& DataSampler::sample_datum(string &yvalue){

  uniform_int_distribution<int> U(0,(this->D-1)*4);
  int idx = U( random_generator );
  int line_idx = (int)( (float)idx / 4 );
  int col_idx = idx % 4;

  //need to fill domain values
  vector<float> probs( domain_values[col_idx].size(),0.0);

  if (YVALUES[line_idx] == string("N")){ //attaches to noun
    float Z = 0;
    for(int i = 0; i < domain_values[col_idx].size();++i){
      probs[i]  = vdistrib(XVALUES[line_idx][0]) 
	          * x1givenv(XVALUES[line_idx][1],XVALUES[line_idx][0]) 
	          * pgivenx1(XVALUES[line_idx][2],XVALUES[line_idx][1])
	          * x2givenx1p(XVALUES[line_idx][3],XVALUES[line_idx][1],XVALUES[line_idx][2]);
      Z += probs[i];
    }
    for(int i = 0; i < domain_values[col_idx].size();++i){ probs[i] /= Z;}
  }
  if (YVALUES[line_idx] == string("V")){ //attaches to verb
    float Z = 0;
    for(int i = 0; i < domain_values[col_idx].size();++i){
      probs[i]  = vdistrib(XVALUES[line_idx][0]) 
	          * x1givenv(XVALUES[line_idx][1],XVALUES[line_idx][0]) 
	          * pgivenv(XVALUES[line_idx][2],XVALUES[line_idx][0])
	          * x2givenvp(XVALUES[line_idx][3],XVALUES[line_idx][0],XVALUES[line_idx][2]);
      Z += probs[i];
    }
    for(int i = 0; i < domain_values[col_idx].size();++i){ probs[i] /= Z;}
  }
  //cumulative distrib
  std::uniform_real_distribution<double> dist(0.0,1.0);
  float r = dist(random_generator);
  float cum_prob = 0;
  for(int i = 0; i < probs.size();++i){
    cum_prob += probs[i];
    if(cum_prob > 1.0 && i != probs.size()-1){cerr << "illegal cum prob : "<< cum_prob << ":" << i << "/"<< probs.size() << endl;}
    if (cum_prob > r){
      XVALUES[line_idx][col_idx] = domain_values[col_idx][i];
      yvalue = YVALUES[line_idx];
      return XVALUES[line_idx];
    }
  }
  //fall back
  XVALUES[line_idx][col_idx] = domain_values[col_idx][probs.size()-1];
  yvalue = YVALUES[line_idx];
  return XVALUES[line_idx];
}


