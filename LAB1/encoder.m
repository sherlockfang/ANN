clear
ndata = 100;
nclass = 8;
epoch = 2000;
Nhidden = 2;
w = normrnd(0,1,Nhidden,nclass+1);
v = normrnd(0,1,nclass+1,Nhidden+1);
eta = 0.001;
h = randi(nclass,ndata,1);
% y is a vector of labels
y_one_hot = -ones( size( h, 1 ), nclass );
% assuming class labels start from one
for i = 1:nclass
    rows = h == i;
    y_one_hot( rows, i ) = 1;
end
y_one_hot = [y_one_hot';ones(1,ndata)];
targets = y_one_hot;
hin = w * y_one_hot;%forward
hout = [2 ./ (1+exp(-hin)) - 1 ; ones(1,ndata)];
oin = v * hout;
out = round(2 ./ (1+exp(-oin)) - 1);
delta_o = (out - targets) .* ((1 + out) .* (1 - out)) * 0.5;%backward
delta_h = (v' * delta_o) .* ((1 + hout) .* (1 - hout)) * 0.5;
delta_h = delta_h(1:Nhidden, :);
dw = delta_h * y_one_hot';
dv = delta_o * hout';
w = w + dw .* eta;
v = v + dv .* eta;
alpha = 0.9;
for j=1:epoch-1
    hin = w * y_one_hot;%forward
    hout = [2 ./ (1+exp(-hin)) - 1 ; ones(1,ndata)];
    oin = v * hout;
    out = round(2 ./ (1+exp(-oin)) - 1);
    delta_o = (out - targets) .* ((1 + out) .* (1 - out)) * 0.5;%backward
    delta_h = (v' * delta_o) .* ((1 + hout) .* (1 - hout)) * 0.5;
    delta_h = delta_h(1:Nhidden, :);
    dw = (dw .* alpha) - (delta_h * y_one_hot') .* (1-alpha);%weight update
    dv = (dv .* alpha) - (delta_o * hout') .* (1-alpha);
    w = w + dw .* eta;
    v = v + dv .* eta;
end
count = 0;
for i=1:ndata
    if out(1:nclass, i) == targets(1:nclass, i)
        count = count + 1;
    end
end
rate = count / ndata;