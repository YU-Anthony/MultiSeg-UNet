imgPath1 = '/CTData/'; 
imgPath2 = '/LabelData/'; 
% source folder
CTDir  = dir([imgPath1 '*.mat']); 
LabelDir  = dir([imgPath2 '*.mat']); 

% save folder
targetCTFolder = '.../synthetic_image/trainCT/';
targetCTFolder2 = '.../synthetic_image/valCT/';

targetLabelFolder = '.../synthetic_image/trainLabel/';
targetLabelFolder2 = '.../synthetic_image/valLabel/';



index = 1;
val_index = 1;

for i = 1:length(CTDir)

    index
    
    alphaValue = randi([40,50],1,4);
    gamaValue = randi([150,210],1,5);
    betaValue =  randi([-30,30],1,6);

    
    CTfileName = CTDir(i).name 
    CT = load([imgPath1,CTfileName]);
    nV = CT.nV;
    
    LabelfileName = LabelDir(i).name; 
    Label = load([imgPath2,LabelfileName]);
    labelV = Label.labelV;
    
    % read data and simulate X-ray
    for j = 1:ceil(length(betaValue)/10*7)
        saveCTPath = [targetCTFolder,num2str(index),'.png'];
        saveLabelPath = [targetLabelFolder,num2str(index),'.png'];
        
        getCT(nV,saveCTPath,210,betaValue(j),180-betaValue(j));
        getLabel(labelV,saveLabelPath,210,betaValue(j),180-betaValue(j));
        
        index = index +1;
    end
    
    for h = ceil(length(betaValue)/10*7)+1:length(betaValue)
        saveCTPath2 = [targetCTFolder2,num2str(val_index),'.png'];
        saveLabelPath2 = [targetLabelFolder2,num2str(val_index),'.png'];
        
        getCT(nV,saveCTPath2,210,betaValue(h),180-betaValue(h));
        getLabel(labelV,saveLabelPath2,210,betaValue(h),180-betaValue(h));
        
        val_index = val_index +1;
    end
end