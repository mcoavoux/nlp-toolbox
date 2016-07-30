#include "LP.h"
#include <fstream>
#include "utils.h"
#include <cassert>
#include <set>
#include <random>

void tokenize_dataline(wstring const& raw_line,vector<wstring> &tokens,wchar_t const *delim){
  
  wstring delimiters(delim);
  tokens.clear();

  int lpos = 0;
  int pos = raw_line.find_first_not_of(delimiters,lpos);
  while (pos != std::wstring::npos) {
    lpos = pos;
    pos = raw_line.find_first_of(delimiters,lpos);
    tokens.push_back(raw_line.substr(lpos,pos-lpos));
    //cout << tokens.back()<<"*"<<endl;
    lpos = pos;
    pos = raw_line.find_first_not_of(delimiters,lpos);
  }
} 

bool isspace(wstring &str){
  if (str.empty()){return true;}
  for(int i = 0; i < str.size();++i){
    if(!std::isspace(str[i])){return false;}
  }
  return true;
}



class sqeuclidean{
public:

  float operator()(float a,float b){
    return pow(a-b,2);
  }  
};

void Proximity::aggregate(vector<float> const &xvec,vector<float> const &yvec,float weight){
  vector<float> aggr(xvec.size(),0.0);
  std::transform (xvec.begin(), xvec.end(), yvec.begin(),aggr.begin(), sqeuclidean());
  dist += std::accumulate(aggr.begin(),aggr.end(),0,std::plus<float>())*weight;
}

float Proximity::proximity(){
  float res =  exp(-dist);
  //float res = dist;
  dist = 0;
  return res;
}


WSDdataset::WSDdataset(char const *datafilename){
  code_dataset(datafilename,vector<unsigned>());
  read_dataset(datafilename);

}

void WSDdataset::code_dataline(vector<wstring> const &fields){

  vector<wstring> xvec(fields.begin()+1,fields.end());
  xdata.push_back(xvec);
  ydata.push_back(fields[0]); 
}

void WSDdataset::code_dataset(char const *datafilename, vector<unsigned> const &dense_idxes)
{
    wifstream infile(datafilename);
    wstring bfr;
    unsigned arity = 0; 
    vector<vector<wstring>> allcolumns;
    if(getline(infile,bfr)){
       vector<wstring> fields;
       tokenize_dataline(bfr,fields,L" \n\t");
       arity = fields.size();
       allcolumns.resize(arity);
       for(int i = 0; i < arity;++i){
	 allcolumns[i].push_back(fields[i]);
       }
    }    
    vector<wstring> fields;    
    while(getline(infile,bfr)){
       tokenize_dataline(bfr,fields,L" \n\t");
       assert(fields.size() == arity);
        for(int i = 0; i < arity;++i){
	 allcolumns[i].push_back(fields[i]);
       }
    }
    infile.close();
    //one hot encoding
    yencodings.clear();
    set<wstring> yset(allcolumns[0].begin(),allcolumns[0].end());
    yset.erase(L"?");
    int i = 0;
    for(set<wstring>::iterator it = yset.begin();it!=yset.end();++it){
      vector<float> y(yset.size(),0.0);
      y[i]=1.0;
      yencodings.set_value(*it,y);
      ++i;
    }

    xencodings.clear();
    xencodings.resize(arity-1);
    for (int k = 1; k < arity;++k){
      set<wstring> xset(allcolumns[k].begin(),allcolumns[k].end());
      int i = 0;
      for(set<wstring>::iterator it = xset.begin();it!=xset.end();++it){
	vector<float> x(xset.size(),0.0);
	x[i]=1.0;
	xencodings[k-1].set_value(*it,x);
	++i;
      }
    }    
}


void WSDdataset::read_dataset(char const *datafilename){

  wifstream infile(datafilename);
  wstring bfr;
  unsigned K = 0;
 
  ydata.clear();
  xdata.clear();

  while (getline(infile,bfr)){
     vector<wstring> fields;
     tokenize_dataline(bfr,fields,L" \n\t");
     code_dataline(fields);
  }
  infile.close();
}


void WSDdataset::get_graph(WSDGraph &graph,Proximity *distfunc){

  vector<vector<float>> ydistribs;   
  //random generator
  random_device rnd_device;
  mt19937 mt(rnd_device());
  std::uniform_real_distribution<float> udist(0.0,1.0);

  unsigned N = ydata.size();
  assert(xdata.size()==N);
  //Y soft values
  unsigned D = yencodings.size();
  unsigned last_labelled_item=0;
  for(int i = 0; i < N ;++i){
    if (yencodings.has_value(ydata[i])){
      ydistribs.push_back(yencodings[ydata[i]]);
      last_labelled_item = i;
    }else{//if not a referenced class like '?', throw a random vector instead
      //
      vector<float> rdmy(D);
      float Z = 0;
      for(int i = 0; i < D;++i){rdmy[i]=udist(mt);Z+=rdmy[i];}
      for(int i = 0; i < D;++i){rdmy[i]/=Z;}
      //vector<float> rdmy(D,(float)1/(float)D);
      ydistribs.push_back(rdmy);
    }
  }

  //X embeddings
  vector<vector<vector<float>>> xcodes;
  for(int i = 0; i < N;++i){
    vector<vector<float>> xcoded;
    for(int j = 0; j < xdata[i].size();++j){
      xcoded.push_back(xencodings[j][xdata[i][j]]);
      vector<float> v = xencodings[j][xdata[i][j]];
    }
    xcodes.push_back(xcoded);
  }
  //Xmatrix
  vector<vector<float>> prox_matrix(N);
  for(int i = 0; i < N;++i){prox_matrix[i].resize(N,0.0);}

  for(int i = 0; i < N;++i){
    for(int j = 0; j < N;++j){
      for(int k = 0; k < xcodes[i].size();++k){
	distfunc->aggregate(xcodes[i][k],xcodes[j][k]);
      }
      prox_matrix[i][j] = distfunc->proximity();
      prox_matrix[j][i] = prox_matrix[i][j];
    }
  }
  for(int i = 0; i < N;++i){
    for(int j = 0; j < N;++j){
    }
  }
  graph.set_labels(ydistribs,last_labelled_item);
  graph.set_transitions(prox_matrix);
}


WSDGraph::WSDGraph(vector<vector<float>> const &xmatrix,vector<vector<float>> const &ylabel,unsigned last_labelled_item){

  set_labels(ylabel,last_labelled_item);
  set_transitions(xmatrix);

}


void WSDGraph::set_transitions(vector<vector<float>> const &xmatrix){
  vector<float> flat_matrix;
  for (int i = 0; i < xmatrix.size();++i){
    flat_matrix.insert(flat_matrix.end(),xmatrix[i].begin(),xmatrix[i].end());
  }
  this->X = af::array(xmatrix.size(),xmatrix.size(),flat_matrix.data());
  row_normalize_matrix(X);//note diag != 1 because normalization

}

void  WSDGraph::row_normalize_matrix(af::array &mat){
  mat /= af::tile(af::sum(mat, 1),1,mat.dims(1));
}

void WSDGraph::nullify_diag(float zerovalue){ //sets the reflexive arc weights to 0
  
  af::array D = ! af::identity(X.dims(0),X.dims(1));
  X *= D;
  row_normalize_matrix(X);
}


void WSDGraph::set_labels(vector<vector<float>> const &ylabel,unsigned last_label_idx){

  vector<float> flat_labels(ylabel.size()*ylabel[0].size());
  for (int i = 0; i < ylabel.size();++i){
    for(int j = 0 ;j < ylabel[0].size();++j){
      flat_labels[j*ylabel.size()+i] = ylabel[i][j];
    }
  }
  this->Y = af::array(ylabel.size(),ylabel[0].size(),flat_labels.data());
  row_normalize_matrix(Y);
  this->LN = last_label_idx;
}


void WSDGraph::label_propagation(float min_eps){

  float eps = 100000000000;
  af::array Yref = Y.copy();
  while(min_eps < eps){
    af::array Yb = Y.copy();
    Y = af::matmul(X,Yb);//propag
    Y/= af::tile(af::sum(Y, 1),1,Y.dims(1));//normalize
    Y(af::seq(0,LN),af::span) =  Yref(af::seq(0,LN),af::span);//clamp
    eps = af::sum<float>(af::abs(Y-Yb));//test for convergence
    cout << "Epsilon : "<< eps << endl;
  }
  //af_print(Y);
  //af_print(X);

}



void WSDGraph::knn(unsigned k){
  
  nullify_diag(0);
  unsigned N = X.dims(0);
  af::array mask = af::constant(0,N,N);
  for(int i = 0;i < N;++i){
    af::array sorted_row,IDX;
    af::sort(sorted_row,IDX,X(i,af::span),1,false);
    IDX = IDX(af::seq(k));
    mask(i,IDX) = 1;
  }
  X*= mask;
  row_normalize_matrix(X);
}

void WSDGraph::eps_nn(float eps){
  af::array mask = X > eps;
  X*= mask;
  row_normalize_matrix(X);
}

wostream& WSDGraph::json(wostream &out){
  
  //header
  out << "{\"directed\": false, \"graph\":{},\"multigraph\": false, ";

  //nodes
  wstring node_string = L"\"nodes\": [";
  for(int i = 0; i < Y.dims(0);++i){
    node_string += L"{\"id\": "+to_wstring(i)+ L", \"label\":\""+ to_wstring(i) +L"\", \"sem\": ["+to_wstring(Y(i,0).scalar<float>());
    for(int j = 1; j < Y.dims(1);++j){
      node_string += L","+to_wstring(Y(i,j).scalar<float>());
    }
    //seed
    node_string+= L"] , \"seed\": ";
    if (i < LN){
      node_string+=L"true ";
    }else{
      node_string+=L"false ";
    }  
    node_string += L", ";
    //group (argmax) maybe remove and postprocess ?
    af::array m;
    af::array imax;
    af::max(m,imax,Y(i,af::span));
    node_string += L"\"group\": "+to_wstring(imax(0).scalar<unsigned>()) +L"},";
  }
  node_string.pop_back();
  node_string += L"], ";
  out << node_string;

  //links
  wstring links_string = L"\"links\": [";
  for(int i = 0; i < Y.dims(0);++i){
     for(int j = 0; j < Y.dims(0);++j){
       float weight = X(i,j).scalar<float>();
       if(weight > 0){
	 links_string += L"{\"source\":" + to_wstring(i) + L", \"target\":" + to_wstring(j) + L", \"weight\":" + to_wstring(weight) +L"},";
       }
     }
  }
  links_string.pop_back();
  links_string += L"]}";
  out << links_string<<endl;
  return out;

}


int main(){
    WSDdataset data("/Users/bcrabbe/Desktop/LP.in");
    WSDGraph G;
    Proximity *prox = new Proximity();
    data.get_graph(G,prox);
    delete prox;
    //G.eps_nn(0.01);
    G.knn(3);
    G.label_propagation(0.0000001);
    wofstream out("json.txt");
    G.json(out);
    out.close();
}
