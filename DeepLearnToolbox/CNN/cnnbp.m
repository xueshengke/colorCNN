% net:  cnn
% y:    labels according to data
function net = cnnbp(net, y)
    n = numel(net.layers);
    %   error = label - output
    net.e = y - net.o;
    %  loss function
    net.L = 0.5 * sum(net.e(:) .^ 2) / size(net.e, 2);

    %%  back propagate deltas
    % delta of output layer
    net.od =  - net.e .* (net.o .* (1 - net.o));   %  output delta of sigm
    net.fvd = (net.ffW' * net.od);                 %  feature vector delta
    % gradient of current layer depends on convolutional layer or downsampling layer
    % in downsampling layer, activating function is linear
    if strcmp(net.layers{n}.type, 'c')         %  if last layer is conv
        net.fvd = net.fvd .* (net.fv .* (1 - net.fv));
    end

    %  reshape feature vector deltas into output map style
    sa = size(net.layers{n}.a{1});
    fvnum = sa(1) * sa(2);
    for j = 1 : numel(net.layers{n}.a)
        net.layers{n}.d{j} = reshape(net.fvd(((j - 1) * fvnum + 1) : j * fvnum, :), sa(1), sa(2), sa(3));
    end

    % calculate delta of each layer
    for l = (n - 1) : -1 : 1
        if strcmp(net.layers{l}.type, 'c')
            for j = 1 : numel(net.layers{l}.a)  % delta is divided by number of elements when upsampling
                % vary due to activation function : sigmoid
                net.layers{l}.d{j} = net.layers{l}.a{j} .* (1 - net.layers{l}.a{j}) ...
                    .* (expand(net.layers{l + 1}.d{j}, [net.layers{l + 1}.scale net.layers{l + 1}.scale 1]) / net.layers{l + 1}.scale ^ 2);
            end
        elseif strcmp(net.layers{l}.type, 's')
            for i = 1 : numel(net.layers{l}.a)
                z = zeros(size(net.layers{l}.a{1}));
                for j = 1 : numel(net.layers{l + 1}.a)  % upsampling layer, activating funciton is linear, the gradient is 1
                     z = z + convn(net.layers{l + 1}.d{j}, rot180(net.layers{l + 1}.k{i}{j}), 'full');
                end
                net.layers{l}.d{i} = z;
            end
        end
    end

    %%  calculate gradients
    % gradient of delta with respect to kernel (kernels like weights in
    % traditional neural networks) and bias
    for l = 2 : n
        if strcmp(net.layers{l}.type, 'c')
            for j = 1 : numel(net.layers{l}.a)
                for i = 1 : numel(net.layers{l - 1}.a)
                    net.layers{l}.dk{i}{j} = rot180(convn(net.layers{l - 1}.a{i}, rot180(net.layers{l}.d{j}), 'valid')) / size(net.layers{l}.d{j}, 3);
                end
                net.layers{l}.db{j} = sum(net.layers{l}.d{j}(:)) / size(net.layers{l}.d{j}, 3);
            end
        elseif strcmp(net.layers{l}.type, 's')
           for j = 1 : numel(net.layers{l}.a)
                net.layers{l}.db{j} = sum(net.layers{l}.d{j}(:)) / size(net.layers{l}.d{j}, 3);
           end
        end
    end
    net.dffW = net.od * (net.fv)' / size(net.od, 2);
    net.dffb = mean(net.od, 2);

    function X = rot180(X)
        X = flipud(fliplr(X));
    end
end
