function [ac,err]= dot_autocov(x,y,lagmax)
n = length(x);
lags = -lagmax:1:lagmax;
ac = zeros(1,length(lags));


for ll = 1:length(lags)
    l = lags(ll);
    dot_total = 0;
    dots_ll = [];

    dot_count = 0;
    if l<0
        for i = (-l+1):n
            x_i = x(:,i);
            y_i = y(:,i+l);
            dot_i = dot(x_i, y_i)/(vecnorm(x_i)*vecnorm(y_i));
            dots_ll=[dots_ll;dot_i];
            dot_total = dot_i+dot_total;
            dot_count = dot_count+1;
        end
    else %l>=0
        for i = 1:n-l
            x_i=x(:,i);
            y_i = y(:,i+l);
            dot_i = dot(x_i, y_i)/(vecnorm(x_i)*vecnorm(y_i));
                         dots_ll=[dots_ll;dot_i];
          
            dot_total = dot_i+dot_total;
            dot_count = dot_count+1;
        end

        
    end
    dotmean(ll) = nanmean(dots_ll);
    dotstdev(ll) = nanstd(dots_ll);
    doterr(ll) = nanstd(dots_ll)/sqrt(n-abs(l));
    
    ac(ll) = dot_total/(n-abs(l));%;-abs(l));
    dot_counts(ll) = dot_count/(n-abs(l));%-abs(l));
end

err=doterr;
end
