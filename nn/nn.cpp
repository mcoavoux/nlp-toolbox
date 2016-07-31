#include <arrayfire.h>
#include <vector>
#include <cassert>
#include <unordered_map>
#include <stdexcept>
#include <numeric>

using namespace std;

class UnknownSymbolException: public std::runtime_error{
public:
  UnknownSymbolException(string token):runtime_error((string("Word not found :")+ token)){}
};


class ReLUActivation{

public:

  af::array& predict(af::array const &xinput){
    this->predictions = af::max(0,xinput);
    return this->predictions;
  }

  af::array& backprop(af::array const &xdelta){
    this->xdelta = (xdelta > 0).as(f32);
    return this->xdelta;
  }
  af::array& get_delta(){return this->xdelta;}
  af::array& get_predictions(){return this->predictions;}

private:
  af::array predictions,xdelta;

};


class BiMap{
  /**
   * Bijective (bidirectional map) that allows to encode decode symbols efficiently back and forth
   */
public:

  BiMap(){}

  BiMap(vector<string> const &known_values,string const &unk_val){
    insert_value(unk_val);
    unk_word = true;
    for(int i = 0; i < known_values.size();++i){
          insert_value(known_values[i]);
    }
  }

  BiMap(vector<string> const &known_values){
    unk_word = false;
    for(int i = 0; i < known_values.size();++i){
          insert_value(known_values[i]);
    }
  }
 
  BiMap(BiMap const &other){
    idx2val = other.idx2val;
    val2idx = other.val2idx;
    unk_word = other.unk_word;
  }

  unsigned size()const{return idx2val.size();}

  BiMap& operator= (BiMap const &other){
    idx2val = other.idx2val;
    val2idx = other.val2idx;
    unk_word = other.unk_word;
    return *this;
  }

  void insert_value(string const &value){
    std::unordered_map<std::string,unsigned>::const_iterator got = val2idx.find(value);
    if (got == val2idx.end()){
      idx2val.push_back(value);
      val2idx[value] = idx2val.size()-1;
    }
  }
  
  unsigned operator()(string const &word)const{
    std::unordered_map<std::string,unsigned>::const_iterator got = val2idx.find(word);
    if (got != val2idx.end()){
      return got->second;
    }
    if (unk_word){
      return 0;
    }
    throw UnknownSymbolException(word);
  }

  af::array onehotvector(string const &word){
    af :: array x = af::constant(0,idx2val.size());
    x((*this)(word)) = 1.0;
    return x;
  }

  af::array onehotmatrix(vector<string> const &wordlist){
    af :: array x = af::constant(0,idx2val.size(),wordlist.size());
    for(int i = 0; i < wordlist.size();++i){
      x((*this)(wordlist[i]),i) = 1.0;
    }
    return x;
  }

  unsigned encode(string const &word)const{
    return (*this)(word);
  }  

  void encode_all(vector<string> const &values,vector<unsigned> &codes){
    if (codes.size()==values.size()){
      for(int i = 0; i < values.size();++i){
	codes[i] = (*this)(values[i]);
      }
    }else{
      codes.clear();
      for(int i = 0; i < values.size();++i){
	codes.push_back((*this)(values[i]));
      }
    }
  }
 
  const string& decode(unsigned code)const{
    if (code < idx2val.size()){
      return idx2val[code];
    }
    if (unk_word){
      return idx2val[0];
    }
    throw UnknownSymbolException(to_string(code));
  } 

 private:
  bool unk_word = false;
  vector<string> idx2val;
  unordered_map<string,unsigned> val2idx;
};


//aux function for softmaxlayer
af::array divide(const af::array &a, const af::array &b){
    return a / b;
}



class SoftMaxLoss{
  /**
   * Top Level softmax layer with cross entropy loss integrated in it.
   */
public:

  SoftMaxLoss(vector<string> const &ydictionary){

    ymapper        = BiMap(ydictionary);
    this->Size     = ymapper.size();

  }
  unsigned size()const{return this->Size;}

  float get_loss()const{//assumes ref is already set (backprop has been performed for last prediction.
    return -af::sum<float>(af::log(yout)*yreference);
  }

  af::array& get_predictions(){return yout;}
  af::array& get_delta(){return xdelta;}

  string& predict_symbol(af::array const &xinput){
    //return ymapper.decode(af::argmax(yout));
  }

  af::array& predict(af::array const &xinput){

    this->xinput = xinput;
    //af_print(xinput);
    yout   = af::exp(xinput-tile(max(xinput,0),xinput.dims(0)));//slower but avoids overflows
    //af_print(yout);
    af::array Z =  af::sum(yout, 0);
    //af_print(Z);
    yout = af::batchFunc(yout, Z, divide);
    //af_print(yout);
    return yout;

  }

  af::array& backprop(vector<string> const &ref_values){

    set_reference(ref_values);
    //af_print(yout);
    //af_print(yreference);

    xdelta = yreference - yout;
    return xdelta;
 
  }

protected:

  void set_reference(vector<string> const &ref_values){
    /*
     * Codes the references value for this batch
     */
    yreference = ymapper.onehotmatrix(ref_values);
  }

private:

  BiMap ymapper;

  unsigned Size;
  af::array yout,yreference;
  af::array xinput,xdelta;
};


class LinearLayer{

public:

  LinearLayer(unsigned OutSize, unsigned InSize){
    this->OutSize = OutSize;
    this->InSize = InSize;
    Weights  = 2 * (af::randu(OutSize,InSize) - 0.5);
    bias     = 2 * (af::randu(OutSize)-0.5);
    Jacobian = af::constant(0,OutSize,InSize);
    Jbias    = af::constant(0,OutSize);
  }  

  af::array& get_predictions(){return yout;}
  af::array& get_delta(){return xdelta;}

  af::array& predict(af::array const &xinput){  //batched prediction

    this->xinput = xinput;
    batch_size = xinput.dims(1);
    this->yout   = af::matmul(Weights,xinput) + af::tile(bias,1,batch_size);
    return this->yout;

  }

  af::array& backprop(af::array const &deltaIn){ //batched backprop

    this->Jacobian += af::matmulNT(deltaIn, xinput);
    this->Jbias += af::sum(deltaIn, 1);
    this->xdelta = af::matmulTN(Weights,deltaIn);     
    return xdelta;
  }

  void update(float alpha){                     //performs basic update

    Weights += alpha *Jacobian;
    bias += alpha *Jbias;

    Jacobian = af::constant(0,OutSize,InSize);
    Jbias    = af::constant(0,OutSize);

  }

private:
  unsigned OutSize,InSize;
  unsigned batch_size;
  af::array yout;
  af::array xinput,xdelta;
  af::array Weights, Jacobian;
  af::array bias,Jbias;
};


/**
 * This class encodes a dictionary that can feed embedded discrete symbols to a linear layer
 */
class LookupLayer{

public:

  LookupLayer(unsigned Xsize,unsigned K,vector<string> const& symbols,string const &unk_word){

    embedding_size = K;
    xSize          = Xsize;
    vocab_size     = symbols.size()+1;
    xdictionary    = 2 *(af::randu(embedding_size,vocab_size)-0.5);   //init embeddings randomly
    xmapper        = BiMap(symbols,unk_word);
    jacobian       = af::constant(0,embedding_size,vocab_size);

  }

  //With pretrained external embeddings (first symbol/embedding assumed to stand for unk words
  LookupLayer(unsigned Xsize,
	      af::array const &embeddings,
	      vector<string> const& symbols){

    embedding_size = embeddings.dims(0);
    xSize          = Xsize;
    vocab_size     = embeddings.dims(1);
    xdictionary    = embeddings;  
    string s0      = symbols[0];
    vector<string> sothers = vector<string>(symbols.begin()+1,symbols.end());
    xmapper        = BiMap(sothers,s0);
    jacobian       = af::constant(0,embedding_size,vocab_size);

  }
  af::array& get_predictions(){
    return xembedded;
  }

  const af::array& predict(vector<string> const &xinput){// batch predict
    /*
     * Generates batch_size column vectors, 
     * each of them encoding the concatenation of the embeddings of the input symbols
     */
    assert(xinput.size() % xSize == 0);

    this->batchSize = xinput.size()/xSize;
    xmapper.encode_all(xinput,xcodes);

    af::array vocab_idxes(xcodes.size(),xcodes.data());//shares the indexes with backprop.
    xembedded = af::moddims(xdictionary(af::span,vocab_idxes),xSize*embedding_size,batchSize);

    return xembedded;

  }

  //assumes delta to be a sequence of column vectors
  void backprop(af::array const &delta){               // batch backprop
    af::array M = af::moddims(delta,embedding_size,xcodes.size()); 
    for(int i = 0 ; i < xcodes.size() ; ++i){
      jacobian(af::span,xcodes[i]) += M(af::span,i);//sparse matrix style op.
    }
  }

  void update(float alpha){                     // batch update
    xdictionary += alpha * jacobian;
    jacobian = af::constant(0,embedding_size,vocab_size);
  }


private:
  af::array xembedded;
  af::array xdictionary;
  af::array jacobian;
  vector<unsigned> xcodes;

  unsigned vocab_size;
  unsigned embedding_size;
  unsigned batchSize;   //size of the minibatch
  unsigned xSize; //number of symbols given in each input
  
  BiMap xmapper;
};

class SymbolicClassifier{

public:

  SymbolicClassifier(vector<string> const &yclasses,
		     vector<string> const &xvalues,
		     unsigned xsymbols_size,
		     unsigned hidden_size, 
		     unsigned embedding_size,
		     string unk_word){
    xSize = xsymbols_size;
    output = new SoftMaxLoss(yclasses);    
    output_layer = new LinearLayer(output->size(),hidden_size);
    hidden_layer = new LinearLayer(hidden_size,xsymbols_size*embedding_size);
    input = new LookupLayer(xsymbols_size,embedding_size,xvalues,unk_word);
  }

  void predict(vector<string> const &xvalues){
    //af::timer start1 = af::timer::start();    
    input->predict(xvalues);
    //printf("elapsed seconds (encoding): %g\n", af::timer::stop(start1));
    //af::timer start2 = af::timer::start();
    hidden_layer->predict(input->get_predictions());
    output_layer->predict(hidden_activation.predict(hidden_layer->get_predictions()));
    output->predict(output_layer->get_predictions());
    //printf("elapsed seconds (prediction): %g\n", af::timer::stop(start2));
  }

  void backprop(vector<string> const &yreference){
    //af::timer start1 = af::timer::start();
    output->backprop(yreference);
    output_layer->backprop(output->get_delta());
    hidden_layer->backprop(hidden_activation.backprop(output_layer->get_delta()));
    input->backprop(hidden_layer->get_delta());
    //printf("elapsed seconds (backprop): %g\n", af::timer::stop(start1));
    
  }
  void update(float alpha){
    //af::timer start1 = af::timer::start();
    input->update(alpha);
    hidden_layer->update(alpha);
    output_layer->update(alpha);
    //printf("elapsed seconds (update): %g\n", af::timer::stop(start1));
  }

  void train(vector<string> const &yvalues,
	     vector<string> const &xvalues,
	     unsigned epochs,
	     unsigned batch_size,
	     float alpha){

    unsigned N = yvalues.size();

    assert(xvalues.size() > 0);
    assert(batch_size <= N);
    
    assert(xvalues.size() % N == 0);
    unsigned NB = ceil(yvalues.size()/batch_size);//Num Batches
    vector<unsigned> idxes(N);
    std::iota(idxes.begin(),idxes.end(),0);

    //buffers and slices
    vector<unsigned> batch_idxes(batch_size);
    vector<string> ybatch(batch_size);
    vector<string> xbatch(batch_size*xSize);
    float loss;
    for(int e = 0; e < epochs;++e){
      //cout << "Epoch : " << e << endl;
      loss = 0;
      for (int i = 0; i < NB; ++i){
	batch_idxes.assign(idxes.begin()+ i * batch_size,
			   idxes.begin()+ std::min<unsigned>((i+1) * batch_size,N));
	ybatch.resize(batch_idxes.size());
	xbatch.resize(batch_idxes.size()*xSize);
	for(int j = 0; j < batch_idxes.size();++j){
	  ybatch[j] = yvalues[batch_idxes[j]];
	  std::copy(xvalues.begin()+batch_idxes[j]*xSize,
		    xvalues.begin()+(batch_idxes[j]+1)*xSize,
		    xbatch.begin()+j*xSize);
	}
	predict(xbatch);
	backprop(ybatch);
	update(alpha);
	loss+= output->get_loss();
      }
      cout << "loss : "<< loss/N << endl;
    }
  }

  void test(vector<vector<string>> &xvalues);

private:
  unsigned xSize;
  SoftMaxLoss *output;
  LinearLayer *output_layer;
  LinearLayer *hidden_layer;
  LookupLayer *input;
  ReLUActivation hidden_activation;
};

void test(){

  //Data Type
  vector<string> cls_names = {"V","N","A"};
  vector<string> word_names = {"manger","venir","partir","demeurer","rester","faire","donner","jean","marie","chat","gentil","petit","joli"};  

  //Data Set
  vector<string> yvalues = {"V","V","N","A","V","A","V","V","A","V","V","V","V","V","V","V","V","N","N","N"};
  vector<string> xvalues = {"manger","venir","jean","gentil","manger","petit","partir","demeurer","joli","rester","partir","demeurer","faire","donner","partir","demeurer","rester","marie","jean","chat"};

  /* 
  yvalues.insert(yvalues.end(),yvalues.begin(),yvalues.end());
  xvalues.insert(xvalues.end(),xvalues.begin(),xvalues.end());
  yvalues.insert(yvalues.end(),yvalues.begin(),yvalues.end());
  xvalues.insert(xvalues.end(),xvalues.begin(),xvalues.end());
  */
  SymbolicClassifier cls(cls_names,word_names,1,17,21,string("???"));
  af::timer start1 = af::timer::start();
  cls.train(yvalues,xvalues,100,20,0.001);
  printf("elapsed seconds (overall): %g\n", af::timer::stop(start1));
}


int main(){
  af::info();
  test();
  return 0;
}
