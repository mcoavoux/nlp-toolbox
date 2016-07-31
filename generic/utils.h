#ifndef UTILS_H
#define UTILS_H

#include <unordered_map>
#include <random>
#include <tuple>

using namespace std;

/**
 * Dictionary/mini DB style data structures for processing language data
 */
class KeyNotFoundException : public std::exception{

public:
  KeyNotFoundException(){}
};

//dictionary whose key type must be hashable
template<typename Key,typename Value>
class Dictionary{

public:
  Dictionary();
  Dictionary(Value const &default_val);//dictionary with defaults

  virtual Value& operator[](Key const &key);//generic get/set operator
  virtual const Value& get_value(Key const &key) const;//specific getter 
  virtual bool has_value(Key const &key) const;//specific checker
  virtual void set_value(Key const &key,Value const &value);//specific setter 
  virtual void set_default_value(Value const &value);//dictionary returns this default value when unk key instead of throwing an error
  virtual void unset_default_mode();//dictionary does not return any default anymore => returns error when key not found.
  virtual void keys(vector<Key> &keys)const;
  virtual unsigned size()const{return dict.size();};
  virtual void clear(){dict.clear();};

protected:
  std::unordered_map<Key,Value> dict;
  bool has_default;
  Value default_val;
};


//frequency dictionary whose key type must be hashable
template<typename Key>
class CountDictionary : public Dictionary<Key,unsigned>{

public:
  CountDictionary();
  CountDictionary(unsigned default_val);//dictionary with defaults
  virtual const Key& sample(); //samples a key from the normalized dictionary 
  virtual void add_count(Key const &key,unsigned c = 1.0); //adds c to the total number of counts for this key 

protected:

   bool sampling_allowed;
   std::vector<Key> arg_vec;
   std::discrete_distribution<int> distrib;
   std::default_random_engine rd_generator;

};
 

//dictionary whose values are lists
template<typename Key,typename Value>
class MultiValueDictionary{

public:

     MultiValueDictionary();
     virtual bool has_value(Key const &key) const;//specific checker
     virtual vector<Value>& operator[](Key const &key);//generic get/set operator */
     virtual const vector<Value>& get_value(Key const &key) const;//specific getter  */
     virtual void add_value(Key const &key,Value const &value);//specific setter  */
     Value&  get_values_sum(Key const &key);//aggregates with sum
     Value&  get_values_max(Key const &key);//aggregates with max
     Value&  get_values_mean(Key const &key);//aggregates with mean
     virtual void keys(vector<Key> &keys)const;
     virtual unsigned size()const{return dict.size();};
     virtual void clear(){dict.clear();};

protected: 
  std::unordered_map<Key,vector<Value>> dict;
  
 }; 




///FOR HASHING TUPLES
namespace std{
    namespace
    {

        // Code from boost
        // Reciprocal of the golden ratio helps spread entropy
        //     and handles duplicates.
        // See Mike Seymour in magic-numbers-in-boosthash-combine:
        //     http://stackoverflow.com/questions/4948780

        template <class T>
        inline void hash_combine(std::size_t& seed, T const& v)
        {
            seed ^= hash<T>()(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
        }

        // Recursive template code derived from Matthieu M.
        template <class Tuple, size_t Index = std::tuple_size<Tuple>::value - 1>
        struct HashValueImpl
        {
          static void apply(size_t& seed, Tuple const& tuple)
          {
            HashValueImpl<Tuple, Index-1>::apply(seed, tuple);
            hash_combine(seed, get<Index>(tuple));
          }
        };

        template <class Tuple>
        struct HashValueImpl<Tuple,0>
        {
          static void apply(size_t& seed, Tuple const& tuple)
          {
            hash_combine(seed, get<0>(tuple));
          }
        };
    }

    template <typename ... TT>
    struct hash<std::tuple<TT...>> 
    {
        size_t
        operator()(std::tuple<TT...> const& tt) const
        {                                              
            size_t seed = 0;                             
            HashValueImpl<std::tuple<TT...> >::apply(seed, tt);    
            return seed;                                 
        }                                              

    };
}


#include "utils.hpp"
#endif