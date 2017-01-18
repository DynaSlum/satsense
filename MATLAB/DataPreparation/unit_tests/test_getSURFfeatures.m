% test_getSURFfeatures - testing getSURFfeatures

% load by hand the desired image into rgbI

visualize = true;
class = 'Slum';
class = 'NonBuiltUp';
class = 'BuiltUp';

[ num_points] = getSURFfeatures( rgbI, class, visualize);

disp(['Number of SURF points: ', num2str(num_points)]);