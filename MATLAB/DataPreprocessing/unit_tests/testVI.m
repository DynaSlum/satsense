% script to compute and visualize vegetation mask

vi_type = 'VVI';
vis = false;

path = fullfile('C:','Projects','DynaSlum','Data','Bangalore','GEImages','Clipped');
fname = input('Enter the filename: ','s');
ffname = fullfile(path, fname);

thresh_type =  input('Enter thresholding type [mean|fixed]: ', 's');

rgbROI = imread(ffname);
VI = computeVI(rgbROI, vi_type, vis);
[urbanMask, vegetationMask] = VI2masks(VI, thresh_type, vis);
disp('Vegetation mask');
showMaskOverROI(rgbROI, vegetationMask, thresh_type, 'g');