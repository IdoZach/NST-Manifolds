function [X,Y,Z] = rotatedEllipsoid(center,metric,scale,resolution)
if ~exist('resolution','var')
    resolution = 20;
end
if ~exist('scale','var')
    scale = 0.1;
end
A = metric;
[~, s, rotation] = svd(A);
radii = 1./sqrt(diag(s)) * scale;
u = linspace(0.0, 2.0 * pi, resolution);
v = linspace(0.0, pi, resolution);
x = radii(1) * cos(u)'*sin(v);
y = radii(2) * sin(u)'*sin(v);
z = radii(3) * ones(resolution,1)*cos(v);
X = zeros(resolution);
Y = zeros(resolution);
Z = zeros(resolution);
for i = 1:resolution 
    for j = 1:resolution
        xyz = [x(i,j) y(i,j) z(i,j)] * rotation' + center';
        X(i,j)=xyz(1);
        Y(i,j)=xyz(2);
        Z(i,j)=xyz(3);
    end
end

end