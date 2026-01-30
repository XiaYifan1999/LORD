function plot_err_curve(xin,yin,matches_gt,Gamma)

err =  calc_geo_err_sparse(xin,yin,matches_gt,Gamma);
fprintf('average error: %.4f\n',mean(err))
thresholds = 0:.001:.1;

for j=1:length(thresholds);
     curve(j) = 100*sum(err <= thresholds(j)) / length(err);
end
plot(thresholds,curve,'linewidth',4);

end