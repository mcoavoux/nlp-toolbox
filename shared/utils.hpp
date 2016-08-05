#include <chrono>
#include <numeric>


void tokenize_dataline(string const& raw_line,vector<string> &tokens,char const *delim){
  
  string delimiters(delim);
  tokens.clear();

  int lpos = 0;
  int pos = raw_line.find_first_not_of(delimiters,lpos);
  while (pos != std::string::npos){
    lpos = pos;
    pos = raw_line.find_first_of(delimiters,lpos);
    tokens.push_back(raw_line.substr(lpos,pos-lpos));
    //cout << tokens.back()<<"*"<<endl;
    lpos = pos;
    pos = raw_line.find_first_not_of(delimiters,lpos);
  }
}

bool isspace(string &str){
  if (str.empty()){return true;}
  for(int i = 0; i < str.size();++i){
    if(!std::isspace(str[i])){return false;}
  }
  return true;
}




template<typename Key,typename Value>
Dictionary<Key,Value>::Dictionary(){has_default = false;}

template<typename Key,typename Value>
Dictionary<Key,Value>::Dictionary(Value const &val){set_default_value(val);}


template<typename Key,typename Value>
void Dictionary<Key,Value>::set_default_value(Value const &val){//dictionary returns this default value when unk key instead of throwing an error
   has_default = true;
   default_val = val;
}


template<typename Key,typename Value>
void Dictionary<Key,Value>::keys(vector<Key> &keys)const{

  for(typename std::unordered_map<Key,Value>::const_iterator it=dict.begin();it != dict.end();++it){
    keys.push_back(it->first);
  } 
}

template<typename Key,typename Value>
Value& Dictionary<Key,Value>::operator[](Key const &key) {//generic get / set operator
 
  typename std::unordered_map<Key,Value>::iterator gotit = dict.find(key);

 if (gotit != dict.end()){return gotit->second;}
 if (has_default){
      dict[key] = default_val;
      return dict[key];
 }
 throw KeyNotFoundException();
}

template<typename Key,typename Value>
bool Dictionary<Key,Value>::has_value(Key const &key) const{//specific checker
  typename std::unordered_map<Key,Value>::const_iterator gotit = dict.find(key);
  return gotit != dict.end();
}

template<typename Key,typename Value>
const Value& Dictionary<Key,Value>::get_value(Key const &key) const{
  typename std::unordered_map<Key,Value>::const_iterator gotit = dict.find(key);
  if (gotit != dict.end()){return gotit->second;}
  throw KeyNotFoundException();
}

//specific setter 
template<typename Key,typename Value>
void Dictionary<Key,Value>::set_value(Key const &key,Value const &value){dict[key] = value;}

template<typename Key,typename Value>
void Dictionary<Key,Value>::unset_default_mode(){has_default = false;}

template<typename Key>
CountDictionary<Key>::CountDictionary(){
   this->has_default = false;
   rd_generator = std::default_random_engine(std::chrono::system_clock::now().time_since_epoch().count());
}

template<typename Key>
CountDictionary<Key>::CountDictionary(unsigned def){
  rd_generator = std::default_random_engine(std::chrono::system_clock::now().time_since_epoch().count());
}


template<typename Key>
void CountDictionary<Key>::add_count(Key const &key,unsigned c){ //adds c to the total number of counts for this key 

  typename std::unordered_map<Key,unsigned>::iterator gotit = this->dict.find(key);
  if (gotit != this->dict.end()){
    gotit->second += c;
  }else{
    (*this)[key] = c;
  }
}

template<typename Key>
const Key& CountDictionary<Key>::sample(){

   //put that somewhere else...
   if(!sampling_allowed){
     std::vector<double> w(this->dict.size());
     arg_vec.resize(this->dict.size());
     int idx = 0;
     for(typename std::unordered_map<Key,unsigned>::iterator it = this->dict.begin();it != this->dict.end();++it){
       arg_vec[idx] = it->first;
       w[idx] = it->second;
       ++idx;
     }
     std::discrete_distribution<int> dtmp(w.begin(),w.end());
     distrib.param(dtmp.param()); 
     /*
       std::vector<double> p = distrib.probabilities();
     for(int i = 0; i < p.size();++i){
       std::cout << arg_vec[i] <<  p[i] << endl;
     }
     */

     sampling_allowed = true;
   } 
   //distrib.reset() ?;
   return arg_vec[distrib(rd_generator)];
}
 
template<typename Key,typename Value>
MultiValueDictionary<Key,Value>::MultiValueDictionary(){}

template<typename Key,typename Value>
vector<Value>& MultiValueDictionary<Key,Value>::operator[](Key const &key){
      return dict[key];
}

template<typename Key,typename Value>
const vector<Value>& MultiValueDictionary<Key,Value>::get_value(Key const &key) const{
      return dict[key];
}

template<typename Key,typename Value>
bool MultiValueDictionary<Key,Value>::has_value(Key const &key) const{
  typename std::unordered_map<Key,vector<Value>>::iterator gotit = dict.find(key);
  return (gotit != dict.end());
}


template<typename Key,typename Value>
void MultiValueDictionary<Key,Value>::add_value(Key const &key,Value const &value){
 
  typename std::unordered_map<Key,vector<Value>>::iterator gotit = dict.find(key);
  if (gotit != dict.end()){
    gotit->second.push_back(value);
  }else{
    vector<Value> v;
    v.push_back(value);
    dict[key] = v;
  }
}

template<typename Key,typename Value>
Value&  MultiValueDictionary<Key,Value>::get_values_sum(Key const &key){//aggregates with sum

  typename std::unordered_map<Key,vector<Value>>::iterator gotit = dict.find(key);
  return std::accumulate(gotit->second.begin(),gotit->second.end(),0);

}

template<typename Key,typename Value>
Value&   MultiValueDictionary<Key,Value>::get_values_mean(Key const &key){//aggregates with mean
  typename std::unordered_map<Key,vector<Value>>::iterator gotit = dict.find(key);
  float S = std::accumulate(gotit->second.begin(),gotit->second.end(),0);
  return (S / (float)gotit->second/size());
}

template<typename Key,typename Value>
Value&  MultiValueDictionary<Key,Value>::get_values_max(Key const &key){ //aggregates with max
  typename std::unordered_map<Key,vector<Value>>::iterator gotit = dict.find(key);
  return std::accumulate(gotit->second.begin()+1,gotit->second.end(),*(gotit->second.begin()),std::max<Value>());
}

template<typename Key,typename Value>
void MultiValueDictionary<Key,Value>::keys(vector<Key> &keys)const{
  keys.clear();
  for(typename std::unordered_map<Key,vector<Value>>::const_iterator it=dict.begin();it != dict.end();++it){
    keys.push_back(it->first);
  } 
}
