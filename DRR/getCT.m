function getCT(model,savePath,aValue,bValue,gValue)

[mDims,mCols,mRows] = size(model); 
mCenter = round(([mRows,mCols,mDims]+1)/2);

% Camera imaging range (the default value is equal to the maximum value in size)
cRows = max([mRows,mCols,mDims]);
cCols = max([mRows,mCols,mDims]);
cDims = max([mRows,mCols,mDims]);
% Camera imaging resolution (output size)
rRows = mRows;
rCols = mCols;
rDims = mDims;

% Accuracy (sampling interval)
precison = ([cRows,cCols,cDims]-1)./([cRows,cCols,cDims]-1);
% Camera angle
alpha = aValue;
beta = bValue;
gamma = gValue;

% Turn to radians
alpha = alpha/180*pi;
beta = beta/180*pi;
gamma = gamma/180*pi;
% Initialize mesh
mesh.x = zeros(rRows,rCols,rDims);
mesh.y = zeros(rRows,rCols,rDims);
mesh.z = zeros(rRows,rCols,rDims);

% Mesh generation
% We have (rRows, rCols) rays, and each ray samples rDims points. 
% Therefore, a grid of (rRows, rCols, rDims) is established 
% to store the coordinate position of each sampling point on each ray.

% Calculate mesh
for d = 1:rDims
    for r = 1:rRows
        for c = 2:rCols
            % Initialization
            x = 1+(r-1)*precison(1);
            y = 1+(c-1)*precison(2);
            z = 1+(d-1)*precison(3);
            % {Central rotation}
            % Move to origin
            x = x-mCenter(1);
            y = y-mCenter(2);
            z = z-mCenter(3);
            % X axis rotates counterclockwise
            tmp = [x,y,z]; % Avoid overwriting the original value during rotation
            y = tmp(2)*cos(alpha)-tmp(3)*sin(alpha);
            z = tmp(2)*sin(alpha)+tmp(3)*cos(alpha);
            % Y axis rotates counterclockwise
            tmp = [x,y,z]; % Avoid overwriting the original value during rotation
            x = tmp(1)*cos(beta)+tmp(3)*sin(beta);
            z = -tmp(1)*sin(beta)+tmp(3)*cos(beta);
            % Z axis rotates counterclockwise
            tmp = [x,y,z]; % Avoid overwriting the original value during rotation
            x = tmp(1)*cos(gamma)-tmp(2)*sin(gamma);
            y = tmp(1)*sin(gamma)+tmp(2)*cos(gamma);
            % Move back to center
            mesh.x(r,c,d) = x+mCenter(1);
            mesh.y(r,c,d) = y+mCenter(2);
            mesh.z(r,c,d) = z+mCenter(3);
        end
    end
end

V = zeros(rRows,rCols,rDims);
for d = 1:rDims
    for r = 1:rRows
        for c = 1:rCols
            % Nearest neighbor interpolation
            x = round(mesh.x(r,c,d));
            y = round(mesh.y(r,c,d));
            z = round(mesh.z(r,c,d));
            
            if x>=1&&x<=mRows && y>=1&&y<=mCols && z>=1&&z<=mDims
                V(r,c,d) = model(z,y,x);
            end
        end
    end
end

im = zeros(rRows,rCols);
for r = 1:rRows
    for c = 1:rCols
          rayline = V(r,c,:);
          if sum(rayline)>10
            im(r,c) =sum(rayline)-15;
          else
            im(r,c)=0;
          end
    end
end


im = im/mDims;
im = flip(im, 1);
im = imresize(im, [256, 256]);
im = adapthisteq(im);


imwrite(im,savePath,'png','bitdepth',8);


end
