f = load('learn_metric_data.mat');
%%
figure(1)
clf
hold on
n = size(f.ellipses,1);
%
radii = squeeze(f.ellipses(:,2,:));
radii = std(radii,[],2);
%radii = log(1+f.features(:,2));
colors = 1+round(mat2gray(radii)*255);
jj = jet(256);
colors = jj(colors,:);
tcolors='bkmr';
for i=1:n
    cen = squeeze(f.ellipses(i,1,:));
    rad = squeeze(f.ellipses(i,2,:));
    features = f.features(i,:);
    class = f.class(i,5);
    % [h, kurt, np.mean(coh['logcoh']), np.std(coh['logcoh'])] 
    A = squeeze(f.metric(i,:,:));
    im = squeeze(f.imgs(i,:,:));
    %[x,y,z]=ellipsoid(cen(1),cen(2),cen(3),rad(1),rad(2),rad(3));
    [x,y,z]=rotatedEllipsoid(cen,A,0.05,20);
    %[x,y,z]=visualizeDTrot(A,cen,0.0001);
    
    %surf(x,y,z,'edgecolor','none','facecolor',colors(i,:));
    surf(x,y,z,'edgecolor','none','facecolor',tcolors(class-'a'+1));
    if ~mod(i,10)
    % bottom
        tt=text(cen(1),cen(2),cen(3)-0.01,...
            sprintf('%s:h%1.2f,k%1.2f,c%1.2f',class,features(1),features(2),features(3)));
        %tt=text(cen(1),cen(2),cen(3)-0.01,...
        %    sprintf('%s',class));
        tt.Color=tcolors(class-'a'+1);
        r=0.03;
        ss=surface(cen(1)+[0 1; 0 1]*r, ...
                cen(2)+[0 0; 0 0]*r,...
                cen(3)+[0 0; 1 1]*r/2, ...
            'FaceColor', 'texturemap', 'CData', im(1:end/2,1:end/2));
        colormap gray
        %imagesc(cen(1)+[0 1],cen(2)+[0 1],im)
    end
end
xlabel('x'),ylabel('y'),zlabel('z');
view(0,0);


%
coors = squeeze(f.ellipses(:,1,:));
t = linspace(-0.1,0.1,100);
texts = {'H','K','C1','C2'};
%clf
%hold on 
for i=1:size(f.features,2)
    
    feat = f.features(:,i);
    %feat = -coors(:,1)*1-coors(:,2)*3+coors(:,3)*0.1+0.1;
    colors = 1+round(mat2gray(feat)*255);
    jj = jet(256);
    colors = jj(colors,:);

    mdl = fitlm(coors,feat);
    coeffs =  mdl.Coefficients.Estimate;
    coeffs(2:4) = coeffs(2:4) / norm(coeffs(2:4)) / 20;
    %for k=1:size(coors,1)
    %    plot3(coors(k,1),coors(k,2),coors(k,3),'.','color',colors(k,:))
    %end
    plot3([0 coeffs(2)],[0 coeffs(3)],[0 coeffs(4)],'linewidth',4);
    coeffs = double(coeffs);
    text(coeffs(2),coeffs(3),coeffs(4),texts(i),'fontsize',14);
    %input('')
    %clf
    %hold on
end