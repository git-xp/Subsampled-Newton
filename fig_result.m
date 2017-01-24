% This script plots the results from demo_comparison.m
%
% written by Peng Xu, 2/20/2016
close all;
% plotting
labels = cell(length(results),1);
colors = {'b', 'g', 'r','c','m','k','y','r','b','k','r'};
styles = {'-', '--', ':', '-.','-', '--', ':', '-.',':','-.','-'};

%%
figure(1);
for i = 1:length(results)
    a   = results{i};
    semilogy( abs(a.l-l_opt)/l_opt, 'color', colors{i}, 'linewidth', 2, 'linestyle', styles{i} )
    labels{i} = a.name;
    hold on
end

title([sprintf('%s(%d,%d) -', filename,n,d),'-\lambda=',num2str(lambda)]);  
set(gca,'ygrid','on')
set(gca,'fontsize',20)
ylim([1e-15,1])
ylabel('(f(x)-f*)/f*', 'fontsize', 18)
xlabel('Number of iterations', 'fontsize', 18)
legend(labels, 'location', 'best', 'fontsize', 18)


figure(2);
for i = 1:length(results)
    a   = results{i};
    semilogy( a.t, abs(a.l-l_opt)/l_opt, 'color', colors{i}, 'linewidth', 2, 'linestyle', styles{i} )
    labels{i} = a.name; 
    hold on
end
title([sprintf('%s(%d,%d) -', filename,n,d),'-\lambda=',num2str(lambda)]); 
set(gca,'ygrid','on')
set(gca,'fontsize',20)
ylim([1e-15,1])
ylabel('(f(x)-f*)/f*', 'fontsize', 18)
xlabel('Time (s)', 'fontsize', 18)
legend(labels, 'location', 'best', 'fontsize', 18)

figure(3);
for i = 1:length(results)
    a   = results{i};
    semilogy( a.err, 'color', colors{i}, 'linewidth', 2, 'linestyle', styles{i} )
    labels{i} = a.name;
    hold on
end
title([sprintf('%s(%d,%d) -', filename,n,d),'-\lambda=',num2str(lambda)]); 
set(gca,'ygrid','on')
set(gca,'fontsize',20)
ylim([1e-15,1])
ylabel('||w - w*||_2/||w*||_2', 'fontsize', 18)
xlabel('Number of iterations', 'fontsize', 18)
legend(labels, 'location', 'best', 'fontsize', 18)


figure(4);
for i = 1:length(results)
    a   = results{i};
    semilogy( a.t, a.err, 'color', colors{i}, 'linewidth', 3, 'linestyle', styles{i} )
    labels{i} = a.name;
    hold on
end
title([sprintf('%s(%d,%d) -', filename,n,d),'-\lambda=',num2str(lambda)]); 
set(gca,'ygrid','on')
set(gca,'fontsize',20)
ylim([1e-15,1])
ylabel('||w - w*||_2/||w*||_2', 'fontsize', 18)
xlabel('Time (s)', 'fontsize', 18)
legend(labels, 'location', 'best', 'fontsize', 18)



