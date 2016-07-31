#include <set>
#include <random>
#include <algorithm>

LayerMap::LayerMap(){}
LayerMap::~LayerMap(){
  for (unordered_map<string,NetworkUnit*>::iterator it = name2mem.begin();it != name2mem.end();++it){
    delete it->second;
  }
}

bool LayerMap::has_unitname(string const &name)const{
  return name2mem.find(name) != name2mem.end(); 
}

void LayerMap::add_unit(string const &name,NetworkUnit *layer){name2mem[name] = layer;}
NetworkUnit* LayerMap::get_layer(string const &name){return name2mem.at(name);}
NetworkUnit* LayerMap::operator()(string const &name){return get_layer(name);}



template <class OUTSYM,class INSYM>
void SymbolicFeedForwardNetwork<OUTSYM,INSYM>::add_input_layer(char const *name, InputSymbolicUnit<INSYM> *inlayer){

  inlayer->set_name(string(name));
  if (names2layers.has_unitname(inlayer->get_name())){
    throw InvalidArchException(string("duplicate layer name = ")+inlayer->get_name());
  }
  names2layers.add_unit(inlayer->get_name(),inlayer);
  input_layers.push_back(inlayer);
}

template <class OUTSYM,class INSYM>
void SymbolicFeedForwardNetwork<OUTSYM,INSYM>::add_layer(char const *name,NetworkUnit *layer){

  layer->set_name(string(name));
  if (names2layers.has_unitname(name)){
    throw InvalidArchException(string("duplicate layer name = ")+layer->get_name());
  }
  names2layers.add_unit(layer->get_name(),layer);
}


template <class OUTSYM,class INSYM>
void SymbolicFeedForwardNetwork<OUTSYM,INSYM>::set_output_layer(char const *name,TopSymbolicUnit<OUTSYM> *out_layer){
  out_layer->set_name(string(name));
  if (names2layers.has_unitname(out_layer->get_name())){
    throw InvalidArchException(string("duplicate layer name = ")+out_layer->get_name());
  }
  names2layers.add_unit(out_layer->get_name(),out_layer);
  output_layer = out_layer;
  has_output = true;
}

template <class OUTSYM,class INSYM>
void SymbolicFeedForwardNetwork<OUTSYM,INSYM>::connect_layers(char const *namea,char const *nameb){

  string nameA(namea);
  string nameB(nameb);
  if (! (names2layers.has_unitname(nameA) && names2layers.has_unitname(nameB))){
    throw InvalidArchException(string("attempting to connect non existent layers !"));
  }  
  //connect
  NetworkUnit *parent = names2layers.get_layer(nameA);
  NetworkUnit *child  = names2layers.get_layer(nameB);
  parent->add_child(child);
  child->add_parent(parent);

}

template <class OUTSYM,class INSYM>
void SymbolicFeedForwardNetwork<OUTSYM,INSYM>::compileNetwork(){

  if (!has_output){throw InvalidArchException(string("attempting to build a network without output !"));} 
  if (input_layers.size()==0){throw InvalidArchException(string("attempting to build a network without input !"));} 

  set<string> visited_names;
  deque<NetworkUnit*> agenda(input_layers.begin(),input_layers.end());
  topological_order.clear();

  while (agenda.size() > 0){                                              //computes a topological order on the nodes
    NetworkUnit *current = agenda.front();
    agenda.pop_front();
    if (visited_names.find(current->get_name()) == visited_names.end()){
      visited_names.insert(current->get_name());
      topological_order.push_back(current);
      for(int i = 0; i < current->parent_size();++i){
	NetworkUnit *p = current->get_parent_at(i);
	agenda.push_back(p);
      }
    }
  }
  if (topological_order.size() != names2layers.size()){throw InvalidArchException(string("Attempting to build an invalid network (either cyclic or non connex)."));}
}

template <class OUTSYM,class INSYM>
bool SymbolicFeedForwardNetwork<OUTSYM,INSYM>::check_network_geometry(){

  for(int i = 0; i < topological_order.size();++i){

    NetworkUnit *current = topological_order[i];
    unsigned out_dim = current->out_dimensionality();

    for (int j = 0; j < current->parent_size();++j){
      NetworkUnit *parent = current->get_parent_at(j);
      if (parent->in_dimensionality() != out_dim){
	throw InvalidArchException(string("Dimensions do not match: (") + current->get_name() + string(",") + parent->get_name() + string(")"));
      }
    }
  }
  return true;
}


template <class OUTSYM,class INSYM>
void SymbolicFeedForwardNetwork<OUTSYM,INSYM>::predict_batch(){
  for(int i = 0; i < topological_order.size();++i){
    topological_order[i]->predict();
  }
}

template <class OUTSYM,class INSYM>
void SymbolicFeedForwardNetwork<OUTSYM,INSYM>::backprop_batch(){
  for(int i = topological_order.size()-1; i >= 0;--i){
    topological_order[i]->backprop();
  }
}

template <class OUTSYM,class INSYM>
void SymbolicFeedForwardNetwork<OUTSYM,INSYM>::update_batch(float alpha,bool average,bool adagrad){
  for(int i = topological_order.size()-1; i >= 0;--i){
    topological_order[i]->update(alpha,average,adagrad);
  }
}


template <class OUTSYM,class INSYM>
void SymbolicFeedForwardNetwork<OUTSYM,INSYM>::set_batch_data(vector<OUTSYM> const &ydata,
							      vector<vector<INSYM>> const &xdata){

  output_layer->set_reference(ydata);
  set_batch_data(xdata);

}

template <class OUTSYM,class INSYM>
void SymbolicFeedForwardNetwork<OUTSYM,INSYM>::set_batch_data(vector<vector<INSYM>> const &xdata){

  assert(xdata.size() == input_layers.size());//check that we have data for each input layer
  input_layers[0]->set_input(xdata[0]);

  assert(xdata[0].size() % input_layers[0]->vocab_size() == 0);       //checks that bath data is consistent
  unsigned batch_size = xdata[0].size()/input_layers[0]->vocab_size(); //checks that bath data is consistent
  for (int i = 1; i < input_layers.size();++i){//for additional input types    

    input_layers[i]->set_input(xdata[i]);

    assert(xdata[i].size() % input_layers[i]->vocab_size() == 0);      //checks that batch data is consistent
    assert(batch_size == xdata[0].size()/input_layers[0]->vocab_size());//checks that batch data is consistent
  }
}


template <class OUTSYM,class INSYM>
float SymbolicFeedForwardNetwork<OUTSYM,INSYM>::train_one(float alpha,bool adagrad,bool average){

   if(!is_compiled){
    compileNetwork();
    check_network_geometry();
    is_compiled = true;
  }
  predict_batch(); 
  backprop_batch();
  update_batch(alpha,average,adagrad);  
  return output_layer->get_loss();
}


template <class OUTSYM,class INSYM>
void SymbolicFeedForwardNetwork<OUTSYM,INSYM>::train_all(vector<OUTSYM> const &ydata,
							 vector<vector<INSYM>> const &xdata,
							 unsigned epochs,
							 unsigned mini_batch_size,
							 float alpha,
							 bool adagrad,
							 int average){
  //data consistency checks
  unsigned N = ydata.size();
  assert(mini_batch_size <= N);
  assert(xdata.size() == input_layers.size());
  assert(alpha > 0);

  cout << "Processing " << N << " examples" << endl; 

  for(int i = 0; i < xdata.size();++i){assert(xdata[i].size() == input_layers[i]->vocab_size()*N);}

  //data indexing and shuffle
  vector<unsigned> idxes(ydata.size());
  std::iota(idxes.begin(),idxes.end(),0);
  bool AVG = false;
  for(int e = 0; e < epochs;++e){
    std::shuffle(idxes.begin(), idxes.end(), std::mt19937{std::random_device{}()});
    if(e == average){AVG=true;}
    float loss = 0;
    unsigned NBatches = ceil((float)N/(float) mini_batch_size); 
    
    for (int b = 0 ; b < NBatches ; ++b){
      unsigned bbegin = b*mini_batch_size;                           // CONFIGURE MINIBATCH
      unsigned bend   = std::min<unsigned>((b+1)*mini_batch_size,N);
      unsigned bsize  = bend-bbegin;
      vector<OUTSYM>  ybfr(bsize);                                                           
      vector<vector<INSYM>> xbfr(xdata.size());
      for(int i = 0; i < xdata.size();++i){xbfr[i] = vector<INSYM>(bsize*input_layers[i]->vocab_size());}

      for(int i = bbegin; i < bend;++i){                                 //FILL MINIBATCH
	ybfr[i-bbegin] = ydata[idxes[i]];
	for(int j = 0; j < xbfr.size();++j){
	  unsigned XSIZE = input_layers[j]->vocab_size();
	  for(int k = 0; k < XSIZE;++k){
	    xbfr[j][(i-bbegin)*XSIZE+k] = xdata[j][idxes[i]*XSIZE+k] ;
	  }
	}
      }
      set_batch_data(ybfr,xbfr);                                     // TRAIN ONE
      loss += train_one(alpha,adagrad,AVG);
      //cout << "Epoch " << e << ": processed " << (b+1)*mini_batch_size << " examples" << endl; 
    }
    cout << "Epoch "<< e <<" loss = " << loss/N << endl;
  }
}

template <class OUTSYM,class INSYM>
vector<OUTSYM>& SymbolicFeedForwardNetwork<OUTSYM,INSYM>::predict(vector<vector<INSYM>> const &xdata){

   if(!is_compiled){
    compileNetwork();
    check_network_geometry();
    is_compiled = true;
  }

  set_batch_data(xdata);
  predict_batch(); 
  return output_layer->extract_argmaxes();
}

template <class OUTSYM,class INSYM>
float SymbolicFeedForwardNetwork<OUTSYM,INSYM>::eval(vector<OUTSYM> const &ydata,
						     vector<vector<INSYM>> const &xdata){
  
  vector<OUTSYM> yhat = predict(xdata);
  assert(yhat.size()==ydata.size());
  float S = 0;
  cout << "pred/ref"<<endl;
  for(int i = 0; i < yhat.size();++i){
    //cout << yhat[i] << "/" << ydata[i] << endl;
    S += yhat[i] == ydata[i];
  }
  return S/yhat.size();

}

template <class OUTSYM,class INSYM>
float SymbolicFeedForwardNetwork<OUTSYM,INSYM>::eval_avg(vector<OUTSYM> const &ydata,
							 vector<vector<INSYM>> const &xdata){

  for(int i = 0; i < topological_order.size() ;++i){
    topological_order[i]->swap_weights();
  }
  float e = eval(ydata,xdata);
  for(int i = 0; i < topological_order.size() ;++i){
    topological_order[i]->swap_weights();
  }
  return e;
}


