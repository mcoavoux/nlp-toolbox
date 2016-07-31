#include <sstream>
#include <cassert>
#include <numeric>
#include "nnutils.h"

template <class OUTSYM,class INSYM>
SequentialMiniBatch<OUTSYM,INSYM>::SequentialMiniBatch(unsigned sequence_length,vector<unsigned> const &types_sizes){
    this->seq_length  = sequence_length;  
    this->types_sizes = types_sizes;  
    this->T = 0;
    reset();
}

template <class OUTSYM,class INSYM>
SequentialMiniBatch<OUTSYM,INSYM>::SequentialMiniBatch(SequentialMiniBatch<OUTSYM,INSYM> const &other){
    this->seq_length = other.seq_length;
    this->types_sizes     = other.types_sizes;
    this->yvector = other.yvector;
    this->xvector = other.xvector;
    this->T = other.T;
}


template <class OUTSYM,class INSYM>
void SequentialMiniBatch<OUTSYM,INSYM>::reset(){
   xreset();
   yreset();
   T = 0;
}

template <class OUTSYM,class INSYM>
void SequentialMiniBatch<OUTSYM,INSYM>::xreset(){
    xvector.clear();
    xvector.resize(seq_length);
    for(int i = 0; i < seq_length;++i){xvector[i].resize(types_sizes.size());}
}


template <class OUTSYM,class INSYM>
void SequentialMiniBatch<OUTSYM,INSYM>::yreset(){                         

  yvector.clear();
  eof_seq.clear();
  yvector.resize(seq_length); 
  eof_seq.resize(seq_length);

}

template <class OUTSYM,class INSYM>
void SequentialMiniBatch<OUTSYM,INSYM>::pushYdata(unsigned timestep, OUTSYM const &ysymbol, bool eos){ 

    yvector.at(timestep).push_back(ysymbol); 
    if(eos){eof_seq[timestep].push_back(yvector.at(timestep).size()-1);}
}

template <class OUTSYM,class INSYM>
void SequentialMiniBatch<OUTSYM,INSYM>::add_sequenceXY(vector<vector<INSYM>> const &xvalues,
						       vector<OUTSYM> const &yvalues){

        assert(yvalues.size() == xvalues.size());
        for(int i = 0; i < yvalues.size();++i){
            pushYdata(T,yvalues[i],(T+1) == yvalues.size());
            unsigned K = 0;
            unsigned nextK = types_sizes[K];
            for(int j = 0; j < xvalues[i].size();++j){
                 if (j == nextK){
                    ++K;
                    nextK += types_sizes[K];  
                 } 
                 pushXdata(T,K,xvalues[i][j]);
            }
            ++T;
            if (T == seq_length){T = 0;}
        }
}

template <class OUTSYM,class INSYM>
bool SequentialMiniBatch<OUTSYM,INSYM>::finalize(){//erase the last incomplete data line
   
   while (T > 0){
      yvector[T-1].pop_back();
      if ((T-1) < eof_seq.size() && eof_seq[T-1].size() > 0 && eof_seq[T-1].back() >= yvector[T-1].size()){eof_seq[T-1].pop_back();} 
      for(int k = 0; k < types_sizes.size();++k){
          for(int j = 0; j < types_sizes[k]; ++j){
             xvector[T-1][k].pop_back();
          }
      }
      --T;
   }
   return yvector[0].size() > 0;
}

void read_string_dataset(istream *inFile, vector<XSEQUENCE> &xvalues,vector<YSEQUENCE> &yvalues){

    xvalues.clear();
    yvalues.clear();
    yvalues.push_back(YSEQUENCE()); 
    xvalues.push_back(XSEQUENCE());

    string bfr;
    vector<string> vbfr;
    while(getline(*inFile,bfr)){
       if(isspace(bfr)){
          if(!yvalues.back().empty()){
               yvalues.push_back(YSEQUENCE());
               xvalues.push_back(XSEQUENCE());
          }
          continue;  
       }
       tokenize_dataline(bfr,vbfr);
       yvalues.back().push_back(vbfr.back());
       vbfr.pop_back();
       xvalues.back().push_back(vbfr);
    }
}



template <class OUTSYM,class INSYM>
void SymbolicRecurrentNetwork<OUTSYM,INSYM>::set_batch_data(vector<vector<vector<INSYM>>> const &xvalues, vector<vector<OUTSYM>> const &yvalues,unsigned batch_sequence_length){
     if(!is_compiled){
      compileNetwork();
      check_network_geometry();
      is_compiled = true;
    }
    //** Make batch **//
    assert(xvalues.size() == yvalues.size());
    vector<unsigned> type_sizes;
    for(int i = 0; i < input_layers.size();++i){type_sizes.push_back(input_layers[i]->vocab_size());}

    SequentialMiniBatch<OUTSYM,INSYM> batch(batch_sequence_length,type_sizes);
    for(int i = 0 ; i < yvalues.size() ;++i){
      assert(xvalues[i].size() == yvalues[i].size());
      batch.add_sequenceXY(xvalues[i],yvalues[i]);
    }
    batch.finalize();
    set_batch_data(batch);
}


template <class OUTSYM,class INSYM>
void SymbolicRecurrentNetwork<OUTSYM,INSYM>::set_batch_data(SequentialMiniBatch<OUTSYM,INSYM> const &batch){
     
    if(!is_compiled){
      compileNetwork();
      check_network_geometry();
      is_compiled = true;
    }
   
    //allocate the network
    reset_timestep();
    //1. Re-allocate the unfolding
    for (int i = 0; i < topological_order.size();++i){
        //topological_order[i]->clear();
        topological_order[i]->set_seq_length(batch.get_batch_sequence_length());
    }
    //2. Set registers for training
    for(int i = 0; i < batch.get_batch_sequence_length();++i){ //incorrect you need to move the sequence ahead
       output_layer->set_reference(batch.getYbatch(i));
       for(int k = 0; k < batch.get_num_types();++k){
          input_layers[k]->set_input(batch.getXbatch(i,k));
       }
       next_step();
    }
    reset_timestep();
    //3.init memories and memory boundaries for recurrent units
    for(int i = 0; i < recurrent_connections.size();++i){
      recurrent_connections[i].second->init_input(batch.get_batch_N());
      recurrent_connections[i].second->set_eos(batch.get_eos());
    }
}

template <class OUTSYM,class INSYM>
void SymbolicRecurrentNetwork<OUTSYM,INSYM>::reset_timestep(){//resets the RNN to time step 0

    for (int i = 0; i < topological_order.size();++i){
       topological_order[i]->reset_timestep();
    }   
}

template <class OUTSYM,class INSYM>
void SymbolicRecurrentNetwork<OUTSYM,INSYM>::next_step(){
    for (int i = 0; i < topological_order.size();++i){
       topological_order[i]->next_timestep();
    }   
}

template <class OUTSYM,class INSYM>
void SymbolicRecurrentNetwork<OUTSYM,INSYM>::prev_step(){
    for (int i = 0; i < topological_order.size();++i){
       topological_order[i]->prev_timestep();
    }  
}


template <class OUTSYM,class INSYM>
void SymbolicRecurrentNetwork<OUTSYM,INSYM>::predict_sequence(){
  
    if(!is_compiled){
      compileNetwork();
      check_network_geometry();
      is_compiled = true;
    }
    reset_timestep();
    while(output_layer->has_next_timestep()){
        for(int i = 0; i < topological_order.size();++i){
           topological_order[i]->predict();
        }
        //copy memories
        for(int i = 0; i < recurrent_connections.size();++i){
           recurrent_connections[i].second->set_next_input(recurrent_connections[i].first->get_predictions());
        }
        next_step();
    }
}

template <class OUTSYM,class INSYM>
void SymbolicRecurrentNetwork<OUTSYM,INSYM>::backprop_sequence(){

  prev_step();   
  while(output_layer->has_prev_timestep()){
     for(int i = topological_order.size()-1; i >= 0;--i){
       topological_order[i]->backprop();
     }
     prev_step();
  }
}

template <class OUTSYM,class INSYM>
void SymbolicRecurrentNetwork<OUTSYM,INSYM>::update(float alpha,bool average,bool adagrad){
  for(int i = topological_order.size()-1; i >= 0;--i){
    topological_order[i]->update(alpha,average,adagrad);
  }
}

template <class OUTSYM,class INSYM>
void SymbolicRecurrentNetwork<OUTSYM,INSYM>::add_layer(char const *name,NetworkUnit *layer){

  layer->set_name(string(name));
  if (names2layers.has_unitname(name)){
    throw InvalidArchException(string("duplicate layer name = ")+layer->get_name());
  }
  names2layers.add_unit(layer->get_name(),layer);
}

template <class OUTSYM,class INSYM>
void SymbolicRecurrentNetwork<OUTSYM,INSYM>::add_input_layer(char const *name,InputSymbolicUnit<INSYM> *inlayer){

  inlayer->set_name(string(name));
  if (names2layers.has_unitname(inlayer->get_name())){
    throw InvalidArchException(string("duplicate layer name = ")+inlayer->get_name());
  }
  names2layers.add_unit(inlayer->get_name(),inlayer);
  input_layers.push_back(inlayer);

}

template <class OUTSYM,class INSYM>
void SymbolicRecurrentNetwork<OUTSYM,INSYM>::set_output_layer(char const *name,TopSymbolicUnit<OUTSYM> *out_layer){
  out_layer->set_name(string(name));
  if (names2layers.has_unitname(out_layer->get_name())){
    throw InvalidArchException(string("duplicate layer name = ")+out_layer->get_name());
  }
  names2layers.add_unit(out_layer->get_name(),out_layer);
  output_layer = out_layer;
  has_output = true;
}

template <class OUTSYM,class INSYM>
void SymbolicRecurrentNetwork<OUTSYM,INSYM>::connect_layers(char const *namea,char const *nameb){//parent -> child

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
void SymbolicRecurrentNetwork<OUTSYM,INSYM>::connect_recurrent_layers(char const *source,char const *destination){

  string nameA(source);
  string nameB(destination);
  if (! (names2layers.has_unitname(nameA) && names2layers.has_unitname(nameB))){
    throw InvalidArchException(string("attempting to connect non existent layers !"));
  }  
 //connect
  try{

     LinearActivation *parent = dynamic_cast<LinearActivation*> (names2layers.get_layer(nameA));
     RecurrentLayer *child  = dynamic_cast<RecurrentLayer*>(names2layers.get_layer(nameB));
     if (parent == NULL || child == NULL){throw InvalidArchException(string("attempting to establish invalid recurrent connection !"));}
     recurrent_connections.push_back(make_pair(parent,child));
     child->set_memory_dimension(parent->out_dimensionality());

  }catch(int e){

      throw InvalidArchException(string("attempting to establish invalid recurrent connection !"));

  }
}

template <class OUTSYM,class INSYM>
void SymbolicRecurrentNetwork<OUTSYM,INSYM>::compileNetwork(){

  if (!has_output){throw InvalidArchException(string("attempting to build a network without output !"));} 
  if (input_layers.size()==0){throw InvalidArchException(string("attempting to build a network without input !"));} 

  set<string> visited_names;
  deque<NetworkUnit*> agenda(input_layers.begin(),input_layers.end());
  for(int i = 0; i < recurrent_connections.size();++i){
    agenda.push_back(recurrent_connections[i].second);
  }
  topological_order.clear();
  while (agenda.size() > 0){                                              //computes a topological order on the nodes
    NetworkUnit *current = agenda.front();
    agenda.pop_front();
    if (visited_names.find(current->get_name()) == visited_names.end()){
      visited_names.insert(current->get_name());
      //cout << current->get_name()<<endl;
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
bool SymbolicRecurrentNetwork<OUTSYM,INSYM>::check_network_geometry(){ //assumes topological order has been made (otherwise irrelevant).

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
float SymbolicRecurrentNetwork<OUTSYM,INSYM>::train_one(float alpha,bool adagrad, bool average){

  if(!is_compiled){
    compileNetwork();
    check_network_geometry();
    is_compiled = true;
  }
  predict_sequence(); 
  backprop_sequence();
  update(alpha,average,adagrad); 

  reset_timestep();
  float loss = 0;
  while(output_layer->has_next_timestep()){
        loss += output_layer->get_loss();
        next_step();
  }
  return loss/((float)output_layer->get_seq_length());
}

template <class OUTSYM,class INSYM>
void SymbolicRecurrentNetwork<OUTSYM,INSYM>::train_all(vector<vector<vector<INSYM>>> const &xdata,
						       vector<vector<OUTSYM>> const &ydata,
						       unsigned epochs,
						       unsigned mini_batch_size,
						       unsigned batch_sequence_length,
						       float alpha,
						       bool adagrad,
						       int average){

  unsigned N = ydata.size();
  assert(mini_batch_size <= N);
  assert(xdata.size() == N);
  assert(alpha > 0);

  vector<unsigned> type_sizes = get_input_vocab_sizes();

  cout << "Processing " << N << " examples" << endl; 
  //data indexing and shuffle

  vector<unsigned> idxes(N);
  std::iota(idxes.begin(),idxes.end(),0);
  bool AVG = false;

  for(int e = 0; e < epochs;++e){
    float loss = 0;
    std::shuffle(idxes.begin(), idxes.end(), std::mt19937{std::random_device{}()});
    if(e == average){AVG=true;}
    unsigned NBatches = ceil((float)N/(float) mini_batch_size); 
    for(int b = 0; b < NBatches;++b){ 
      unsigned bbegin = b*mini_batch_size;                         
      unsigned bend   = std::min<unsigned>((b+1)*mini_batch_size,N);
      unsigned bsize  = bend-bbegin;
      SequentialMiniBatch<OUTSYM,INSYM> mbatch(batch_sequence_length, type_sizes);
      for(int i = bbegin ; i < bend ; ++i){mbatch.add_sequenceXY(xdata[idxes[i]],ydata[idxes[i]]);}
      if (mbatch.finalize()){
         set_batch_data(mbatch);
         loss += train_one(alpha,adagrad,AVG);
      }else{cerr << "Warning (minibatch too small and trashed). Changing the minibatch size should fix it."<<endl;}
    }
    cout << "Epoch "<< e <<" loss = " << loss/N << endl; //check the loss computations everywhere wrt averaging
  }
}

template <class OUTSYM,class INSYM>
vector<unsigned> SymbolicRecurrentNetwork<OUTSYM,INSYM>::get_input_vocab_sizes()const{
   vector<unsigned> type_sizes ;
   for(int i = 0; i < input_layers.size();++i){type_sizes.push_back(input_layers[i]->vocab_size());}
   return type_sizes;
}


template <class OUTSYM,class INSYM>
float SymbolicRecurrentNetwork<OUTSYM,INSYM>::eval(vector<vector<vector<INSYM>>> const &xvalues, vector<vector<OUTSYM>> const &yvalues,unsigned batch_sequence_length){

   vector<unsigned> type_sizes = get_input_vocab_sizes();

   SequentialMiniBatch<OUTSYM,INSYM> batch(batch_sequence_length,type_sizes);
   for(int i = 0 ; i < yvalues.size() ; ++i){
     batch.add_sequenceXY(xvalues[i],yvalues[i]);
   }
   batch.finalize();
   set_batch_data(batch);
   predict_sequence();  
   reset_timestep();
   unsigned t = 0;
   float correct = 0;
   float total = 0;
   while(output_layer->has_next_timestep()){
      vector<OUTSYM> yhat = output_layer->extract_argmaxes(); 
      vector<OUTSYM> yref = batch.getYbatch(t); 
      assert(yhat.size()==yref.size());
      for(int i = 0; i < yhat.size();++i){
         correct += (yhat[i]==yref[i]);
      }
      total += yhat.size();
      next_step();
     ++t;
   }
   return correct/total;
}

template <class OUTSYM,class INSYM>
float SymbolicRecurrentNetwork<OUTSYM,INSYM>::eval_avg(vector<vector<vector<INSYM>>> const &xvalues, vector<vector<OUTSYM>> const &yvalues,unsigned batch_sequence_length){

  for(int i = 0; i < topological_order.size() ;++i){
    topological_order[i]->swap_weights();
  }
  float e = eval(xvalues,yvalues,batch_sequence_length);
  for(int i = 0; i < topological_order.size() ;++i){
    topological_order[i]->swap_weights();
  }
  return e;
}
