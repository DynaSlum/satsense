clear;

data_dir = '/home/bweel/Documents/projects/dynaslum/data/Bangalore/GE_Images/fixed';
target_dir = '/home/bweel/Documents/projects/dynaslum/data/Bangalore/masks';

files = {
    fullfile(data_dir, 'Bangalore_ROI1.tif')
    fullfile(data_dir, 'Bangalore_ROI2.tif')
    fullfile(data_dir, 'Bangalore_ROI3.tif')
    fullfile(data_dir, 'Bangalore_ROI4.tif')
    fullfile(data_dir, 'Bangalore_ROI5.tif')
};

masks = {
    fullfile(target_dir, 'Bangalore_ROI1_slumMask.tif')
    fullfile(target_dir, 'Bangalore_ROI2_slumMask.tif')
    fullfile(target_dir, 'Bangalore_ROI3_slumMask.tif')
    fullfile(target_dir, 'Bangalore_ROI4_slumMask.tif')
    fullfile(target_dir, 'Bangalore_ROI5_slumMask.tif')
};

for f = 1:size(files,1)
    disp(strcat('Pre-processing file nr ', num2str(f), ': ', files(f,:)));
    file = files(f,:);
    
    [X, R] = geotiffread(cell2mat(file));
    
    % Calculate the vegetation index
    vi = computeVI(X, 'VVI', false);
  
    [urbanMask, vegetationMask] = VI2masks(vi, 'fixed', false, 25, 'VVI');
    
    [slumMask, R2] = geotiffread(cell2mat(masks(f,:)));
    slumMask = slumMask == 1; % make it a logical array
    
    urbanMask = max(urbanMask - slumMask, 0);
    vegetationMask = max(vegetationMask - slumMask, 0);
   
    geotiffwrite(fullfile(target_dir, strcat('Bangalore_ROI', num2str(f), '_urbanMask.tif')), urbanMask, R);
    geotiffwrite(fullfile(target_dir, strcat('Bangalore_ROI', num2str(f), '_vegetationMask.tif')), vegetationMask, R);
end