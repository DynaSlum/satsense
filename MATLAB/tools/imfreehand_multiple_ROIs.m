%% set upmask size
cl;
importfile('C:\Projects\DynaSlum\Data\Kalyan\Rasterized_Lourens\Mumbai_P4_R1C1_3_clipped.tif');
image_data = Mumbai_P4_R1C1_3_clipped;
sz =  [size(image_data,1) size(image_data,2)];

clear Mumbai_P4_R1C1_3_clipped;
totMask = false( sz ); % accumulate all single object masks to this one

%% display the figure image data
figure;
h_im = imshow(image_data);
axis on, grid on;

%% draw muliple ROIs
h = imfreehand( gca ); setColor(h,'red');
position = wait( h );
BW = createMask( h );
while sum(BW(:)) > 10 % less than 10 pixels is considered empty mask
      totMask = totMask | BW; % add mask to global mask
      % you might want to consider removing the old imfreehand object:
      %delete( h ); % try the effect of this line if it helps you or not.

      % ask user for another mask
      h = imfreehand( gca ); setColor(h,'red');
      position = wait( h );
      BW = createMask( h );
end
% show the resulting mask
figure; imshow( totMask ); title('multi-object mask');