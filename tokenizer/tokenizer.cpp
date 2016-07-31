#include "tokenizer.h"
#include "utf8.h"
#include <iostream>
#include <fstream>
#include <getopt.h>


void Tokenizer::set_line(string const &line){
  
  buffer.clear();
  utf8::utf8to32(line.begin(), line.end(), back_inserter(buffer));
  normalize(buffer); 
  bidx = buffer.begin();

}

bool Tokenizer::next_token(wstring &token){return next_token(buffer,token);}

bool Tokenizer::next_token(string &token){

  wstring wtoken;
  bool found = next_token(buffer,wtoken);
  if(!found){return found;}
  token.clear();
  utf8::utf32to8(wtoken.begin(),wtoken.end(), back_inserter(token));
  return found;

}


bool Tokenizer::next_token(wstring const &buffer,wstring &token){  

  //assign the next token (as UTF32) found in the line and returns false when eof is reached. 
  token.clear();
  boost::match_results<wstring::const_iterator> what;

  //A. skips leading wsp
  bool found = boost::regex_search(bidx,buffer.end(),what,wsp_regex);
  //if (!found){return false;}
  bidx = bidx + what.length();

  //B. attempts to match strong compound
  bool cpd_found = boost::regex_search(bidx,buffer.end(),what,scpd_regex);
  if (cpd_found){//removes whitespaces
    int idx = std::distance(buffer.begin(), bidx);
    int end = idx+what.length();
    token.reserve(end-idx);
    for(; idx < end;++idx){
      if(buffer[idx] != L' '){
	token.push_back(buffer[idx]);
      }
    }
    bidx += what.length();
    return true;
  }
  //C. attempts to match weak compound
  cpd_found = boost::regex_search(bidx,buffer.end(),what,wcpd_regex);
  if (cpd_found){//subs whitespaces by '_'
    int idx = std::distance(buffer.begin(), bidx);
    int end = idx+what.length();
    token.resize(end-idx);
    int copy_idx = 0;
    for(; idx < end;++idx){
      if(buffer[idx] == L' '){token[copy_idx] = L'_';}
      else{token[copy_idx] = buffer[idx];}
      ++copy_idx;
    }
    bidx += what.length();
    return true;
  }
  //D. matches next regular word otherwise
  found = boost::regex_search(bidx,buffer.end(),what,full_regex);
    if(found){
      int idx = std::distance(buffer.begin(), bidx);
      token = buffer.substr(idx,what.length());
      bidx = bidx + what.length();
    }
    return found;

}

wstring Tokenizer::make_dictionary_regex(const char *filename){

  ifstream infile(filename);
  if(!infile){cerr << "Error wrong dictionary filename (aborting)"<< endl;exit(1);}
  string bfr;
  wstring regex(L"(");
  boost::wregex sep(L"\\+");   
  while(getline(infile,bfr)){
    wstring wbfr;
    utf8::utf8to32(bfr.begin(), bfr.end(), back_inserter(wbfr));
    regex += boost::regex_replace(wbfr,sep,L" ");
    regex.erase(regex.find_last_not_of(L" \t\f\v") + 1);
    regex+=L"|";
  }
  regex.pop_back();
  regex+=L")"; 
  infile.close();
  return regex;

}

void  Tokenizer::add_strong_cpd_dictionary(string const &dict_name,const char *filename){
//compounds whose components are concatenated
  scpd_dictionaries.push_back(make_dictionary_regex(filename));
}


void Tokenizer::add_weak_cpd_dictionary(string const &dict_name,const char *filename){
  wcpd_dictionaries.push_back(make_dictionary_regex(filename));
}


void Tokenizer::add_prefix_dictionary(string const &dict_name,const char *filename){
//prefixes followed by hyphens followed by words

  ifstream infile(filename);
  if(!infile){cerr << "Error wrong dictionary filename (aborting)"<< endl;exit(1);}
  string bfr;
  wstring regex(L"(");
  boost::wregex sep(L"\\+");   
  while(getline(infile,bfr)){
    wstring wbfr;
    utf8::utf8to32(bfr.begin(), bfr.end(), back_inserter(wbfr));
    regex += boost::regex_replace(wbfr,sep,L" ");
    regex.erase(regex.find_last_not_of(L" \t\f\v") + 1);
    regex += L" [^ \\n\\s\\t]+";//any word can follow the sequence'prefix -'
    regex+=L"|";
  }
  regex.pop_back();
  regex+=L")"; 
  infile.close();
  scpd_dictionaries.push_back(regex);

}


wchar_t translate_char(const wchar_t c){  

  switch (c){
    case L'“':
    case L'”':
    case L'»':
    case L'«':
        return L'"';
    case L'‘':
    case L'’':
        return L'\'';
    default :
        return c;
  }
}


void translate_string(wstring &str){
  for (int i = 0; i < str.size();++i){str[i] = translate_char(str[i]);}
}

void Tokenizer::normalize(wstring &bfr){

  //1. Normalize special chars
  //fait un tr sur les codes de certains guillemets
  translate_string(bfr);

  //2. Normalize spaces around ponct
  bfr = boost::regex_replace(bfr,norm_ponct_regex,L" $& ");//wsp around ponct
  bfr = boost::regex_replace(bfr,norm_wsp_regex,L" ");//one wsp between tokens
  bfr = boost::regex_replace(bfr,norm_apos_regex,L"'");//apostrophes on left token

  if(bfr.front() == L' '){bfr.erase(0,1);} //removes any remaining leading wsp
  if(bfr.back() == L' '){bfr.pop_back();} //removes any remaining trailing wsp
  //wcout << bfr << endl;
}

void Tokenizer::compile(){

  //basic normalisation regexes
  norm_wsp_regex   = boost::wregex(L"([ \\n\\s\\t])+",boost::regex::optimize);//spaces
  norm_ponct_regex = boost::wregex(L"[\\?!\\{;,\\}\\.:/=\\+\\(\\)\\[\\]\"'\\-…]",boost::regex::optimize);//split all poncts agressively 
  norm_apos_regex  = boost::wregex(L" '",boost::regex::optimize);//apostrophes
  
  //segmentation core regexes
  wsp_regex = boost::wregex(L"([ \\n\\s\\t])*",boost::regex::optimize);//wsp
  full_regex = boost::wregex(L"(^[^ \\n\\s\\t]+)",boost::regex::optimize);//basic

  //weak compounds
  wstring wcpd_expr(L"^(");
  for(int i = 0; i < wcpd_dictionaries.size();++i){
     wcpd_expr += wcpd_dictionaries[i]+L"|";
  }
  wcpd_expr.pop_back();
  wcpd_expr += L")";
  if (debug){wcerr << L"Weak compound regex:" << endl << wcpd_expr << endl << endl;}
  wcpd_regex = boost::wregex(wcpd_expr,boost::regex::optimize);

  //strong compounds (also includes dates and numbers)
  wstring scpd_expr(L"^(");
  for(int i = 0; i < scpd_dictionaries.size();++i){
     scpd_expr += scpd_dictionaries[i]+L"|";
  }

  scpd_expr += L"([0-9]+( ([/,\\.])? ?[0-9]+)+)"; //numbers/dates sub expression 
  scpd_expr += L"|([0-9]+( (ème|eme|er|e|è) ))";//(partially including ordinals)
  scpd_expr += L"|( ?\\-)+";                         //hyphens
  scpd_expr += L"|([A-ZÊÙÈÀÂÔÎÉÁ] (\\. )?)+";      //sigles et acronymes (?)
  scpd_expr += L")";
  if (debug){wcerr << L"Strong compound regex:" << endl << scpd_expr << endl << endl;}
  scpd_regex = boost::wregex(scpd_expr,boost::regex::optimize);
}

void Tokenizer::batch_tokenize_file(const char *src_filename,const char *dest_filename,bool column){

  ifstream infile(src_filename);
  if(!infile){cerr << "Error wrong input filename (aborting)"<< endl;exit(1);}
  ofstream outfile(dest_filename);
  string bfr;
  string res;
  while(getline(infile,bfr)){
    //cerr << bfr << endl;
     if(bfr.empty() || bfr.find_first_not_of(' ') == std::string::npos){continue;}//aborts if empty or white space
     set_line(bfr);
     if(!column && next_token(res)){outfile << res;}
     while(next_token(res)){
       if (column){outfile << res << endl;}
       else{outfile << " " << res ;}
     }
     outfile << endl;
  }
  infile.close();
  outfile.close();

}

