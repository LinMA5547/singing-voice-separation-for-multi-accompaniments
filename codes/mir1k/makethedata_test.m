function data_cell_com = makethedata_test( data_cell1,data_cell2)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

  data_cell_com = cell(1,1);


  inputdim = 1026;

      data1 = data_cell1{1,1};data2 = data_cell2{1,1};
      T =size(data_cell1{1,1},1) / inputdim;
	
      for t = 1:T
         
          
                hAct{t} = [data1((t-1)*inputdim+1:t*inputdim);data2((t-1)*inputdim+1:t*inputdim)]; 
          
          

%             inputdim*t - (t-1)*inputdim
%             size(hAct{t})

          data_cell_com{1,1}(2*(t-1)*inputdim+1:2*t*inputdim,1) = hAct{t}';
      end
  
end
          


