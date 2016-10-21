% net:   cnn trained
% x:     train data
% y:     train label
% opts:  parameters
function net = relucnntrain(net, x, y, opts)
    % sample number
    imageNum = size(x, 4);           
    % epoch of each batchsize
    numbatches = imageNum / opts.batchsize;    
    if rem(numbatches, 1) ~= 0
        error('numbatches not integer');
    end
    % save value of cost
    net.rL = zeros(opts.numepochs * numbatches, 1);
    
    for i = 1 : opts.numepochs
        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)]);
        %tic;
        % imageNum is integer randomly
        randIndex = randperm(imageNum);
        for l = 1 : numbatches
            batch_x = x(:, :, :, randIndex((l - 1) * opts.batchsize + 1 : l * opts.batchsize));    
            batch_y = y(:,       randIndex((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            % compute the output of cnn, feedforward
            net = relucnnff(net, batch_x);
            % compute the delta of cnn, back propagation
            % obtain gradients respect to weights and biases
            net = relucnnbp(net, batch_y);
            % update the value of weights and biases, using the gradients
            net = cnnapplygrads(net, opts);
            
            net.rL((i - 1) * opts.numepochs + l) = net.L;
            if net.L < opts.lowThreshold
                disp('train process stops due to lower threshold');
                return ;
            end
        end
        %toc;
        opts.alpha = opts.alpha * 0.993 ;
    end
    
end
