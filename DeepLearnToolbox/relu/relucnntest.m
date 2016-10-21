function [ratio, er, bad] = relucnntest(net, x, y)
    %  feedforward
    net = relucnnff(net, x);
    [~, h] = max(net.o);
    [~, a] = max(y);
    correct = find(h == a);
    bad = find(h ~= a);
    ratio = numel(correct) / size(y, 2);
    er = numel(bad) / size(y, 2);
end
