% net:   cnn 
% x:     part of sample , batches
function net = relucnnff(net, x)
    n = numel(net.layers);
    inputmaps = net.inputmaps ;   % gray = 1, color(RGB) = 3
    for i = 1 : inputmaps
        net.layers{1}.a{i} = reshape(x(:, :, i, :), size(x, 1), size(x, 2), size(x, 4));
    end
    
    for l = 2 : n   %  for each layer
        if strcmp(net.layers{l}.type, 'c')
            %  !!below can probably be handled by insane matrix operations
            for j = 1 : net.layers{l}.outputmaps   %  for each output map
                %  create temp output map
                %  initiate all to zero
                z = zeros(size(net.layers{l - 1}.a{1}) - [net.layers{l}.kernelsize - 1 net.layers{l}.kernelsize - 1 0]);
                for i = 1 : inputmaps   %  for each input map
                    %  convolve with corresponding kernel and add to temp output map
                    z = z + convn(net.layers{l - 1}.a{i}, net.layers{l}.k{i}{j}, 'valid');
                end
                %  add bias, pass through nonlinearity
                net.layers{l}.z{j} = z + net.layers{l}.b{j};
                net.layers{l}.a{j} = relu( net.layers{l}.z{j} );
            end
            %  set number of input maps to this layers number of outputmaps
            inputmaps = net.layers{l}.outputmaps;
            
        elseif strcmp(net.layers{l}.type, 's')
            %  downsample mean pooling
            for j = 1 : inputmaps
%                z = convn(net.layers{l - 1}.a{j}, ones(net.layers{l}.scale) / (net.layers{l}.scale ^ 2), 'valid');   %  !! not the most efficient
%                net.layers{l}.a{j} = z(1 : net.layers{l}.scale : end, 1 : net.layers{l}.scale : end, :);
                mapDim = size(net.layers{l - 1}.a{j});
                z = zeros(mapDim(1) / net.layers{l}.scale, mapDim(2) / net.layers{l}.scale, mapDim(3));
                for row = 1 : net.layers{l}.scale
                    for column = 1 : net.layers{l}.scale
                        z = z + net.layers{l - 1}.a{j}(row : net.layers{l}.scale : end, column : net.layers{l}.scale : end, :);
                    end
                end
                net.layers{l}.a{j} = z ./ (net.layers{l}.scale ^ 2);          
            end
        end
    end

    %  concatenate all end layer feature maps into vector
    %  construct map of last layer into a column vector
    net.fv = [];
    for j = 1 : numel(net.layers{n}.a)
        sa = size(net.layers{n}.a{j});
        net.fv = [net.fv; reshape(net.layers{n}.a{j}, sa(1) * sa(2), sa(3))];
    end
    % feedforward into output perceptrons
    % check the rows and columns of matrice are suitable
    % activation of last layer
    net.o = sigm(net.ffW * net.fv + repmat(net.ffb, 1, size(net.fv, 2)));
end
