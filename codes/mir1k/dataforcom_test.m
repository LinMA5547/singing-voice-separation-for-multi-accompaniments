function cominput = dataforcom_test( modelname,data_cell)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
load(modelname);
[stack,W_t] = rnn_params2stack(theta, eI);
  cominput = cell(1,1);

  
      T =size(data_cell{1,1},1) / eI.inputDim;
	hAct = cell(numel(eI.layerSizes)-1,T);
    
      for t = 1:T
          
         data = data_cell{1,1}((t-1)*eI.inputDim+1:t*eI.inputDim,1);
          for l = 1:numel(eI.layerSizes)-1
            if l == 1
                hAct{1,t} = stack{1}.W * data;
            else
                hAct{l,t} = stack{l}.W * hAct{l-1,t};
            end;
             hAct{l,t} = bsxfun(@plus, hAct{l,t}, stack{l}.b);
           
             if t > 1
                if isfield(eI, 'fullRNN') && eI.fullRNN==1
                    hAct{l,t} = hAct{l,t} + W_t{l}.W * hAct{l,t-1};
                elseif l == eI.temporalLayer
                    hAct{l,t} = hAct{l,t} + W_t * hAct{l,t-1};
                end
            end;
              if strcmpi(eI.activationFn,'tanh')
                hAct{l,t} = tanh(hAct{l,t});
            elseif strcmpi(eI.activationFn,'logistic')
                hAct{l,t} = 1./(1+exp(-hAct{l,t}));
            elseif strcmpi(eI.activationFn,'RELU')
                hAct{l,t} = max(0,hAct{l,t});
            else
                error('unrecognized activation function: %s',eI.activationFn);
            end;
            
          end
          hAct{numel(eI.layerSizes),t} = bsxfun(@plus, stack{numel(eI.layerSizes)}.W * hAct{l,t}, stack{numel(eI.layerSizes)}.b);
          cominput{1,1}((t-1)*eI.layerSizes(end)+1:t*eI.layerSizes(end),1) = hAct{numel(eI.layerSizes),t}';
      end
  
end
          


