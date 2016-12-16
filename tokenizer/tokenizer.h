#include <string>
//#include <regex>
#include <boost/regex.hpp>
using namespace std;

/**
 * A (french) tokenizer operating on UTF-8 encoding.
 * uses a PTB normalisation (wrt clitics and hyphens)
 * includes cpd dictionaries (weak and strong)
 * Performs tokenization with an incremental max match constrained by spaces 
 */
class Tokenizer{

public:

  Tokenizer(bool debug=false){this->debug=debug;};
  void compile();//compile all the regexes before running
  bool next_token(wstring &token);   //assign the next token (as UTF32) found in the line and returns false when eof is reached.
  bool next_token(string &token);   //assign the next token (as UTF8) found in the line and returns false when eof is reached.
  //void batch_tokenize_file(const char *src_filename,const char *dest_filename,bool column=false);
  void batch_tokenize_file(const char *src_filename,const char *dest_filename,bool column=false,bool eos=false,bool parag_break=false);

  void set_line(string const &line);//sets the next UTF8 line to tokenize
  void add_weak_cpd_dictionary(string const &dict_name,const char *filename);//compounds whose components are separated with "_"
  void add_strong_cpd_dictionary(string const &dict_name,const char *filename);//compounds whose components are concatenated
  void add_prefix_dictionary(string const &dict_name,const char *filename);//prefixes followed by hyphen followed by words
  
  //provides an educated guess on the end of sentence status of a token
  bool is_eos(wstring const &token)const; 
  bool is_eos(string const &token)const; 

 protected:
  void normalize(wstring &str);
  wstring make_dictionary_regex(const char *filename);
  bool next_token(wstring const &buffer, wstring &token);   //assign the next token (as UTF32) found in the line and returns false when eof is reached.

private:
  bool debug;
  wstring buffer;
  std::wstring::const_iterator bidx;
  vector<wstring> wcpd_dictionaries;
  vector<wstring> scpd_dictionaries;

  //Normalization regexes
  boost::wregex norm_wsp_regex;
  boost::wregex norm_ponct_regex;
  boost::wregex norm_apos_regex;//glues apostrophes to the leftmost word
  
  //segmentation regexes
  boost::wregex wsp_regex;//skips leading white space during segmentation
  boost::wregex wcpd_regex;//weak compound insert '_' between components
  boost::wregex scpd_regex;//strong compound concatenate components

  boost::wregex full_regex;//matches next token
};
