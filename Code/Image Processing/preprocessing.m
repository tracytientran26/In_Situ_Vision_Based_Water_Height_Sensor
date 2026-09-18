% get access to the newest image
% eyeOfHorusPi = raspi("127.0.1.1", "pi", "raspberry");
% folderAddress = system(eyeOfHorusPi, "home/tracytran/Desktop/Camera_Image");
% newestUpdate = system(eyeOfHorusPi, "ls -t folderAddress | head -n 1");
% newestUpdateAddress =  system(eyeOfHorusPi, ...
%    "home/tracytran/Desktop/Camera_Image" + newestUpdate);
% importedImage = getFile(eyeOfHorusPi, newestUpdate);

% get access to laptop's images
folderAddress ="C:\Users\tranp\OneDrive\Desktop\ARTS lab\Camera Project\Images";
fileAddress = "Creek image 2_1403.bmp";
get_img = fullfile(folderAddress, fileAddress);

% import to matlab
img = imread(get_img);
imshow(img);
title('img');
figure;

% Convert the image to grayscale for further processing
gray_img = im2gray(img);
imshow(gray_img);
title('gray_img');
figure;
%adj_img = imadjust(gray_img);
binary_img = im2double(gray_img);
Ieq = adapthisteq(binary_img);
imshow(Ieq);
title('Ieq');
figure;

% call mask
[BW, maskedImage] = segmentImage(Ieq);
imshow(BW);
title('BW');
figure;
imshow(maskedImage);
title('maskedImage');
figure;

% get pixels
water_pixels = nnz(maskedImage);
total_pixels = numel(maskedImage);

% Calculate the percentage of water pixels in the masked image
water_percentage = (water_pixels / total_pixels) * 100;

% Display the calculated percentage of water pixels
fprintf('Percentage of water pixels: %.2f%%\n', water_percentage);

% Save the masked image showing water areas
outputFileName = fullfile(folderAddress, 'maskedImage.png');
imwrite(maskedImage, outputFileName);
fprintf('Masked image saved as: %s\n', outputFileName);

% Calculate the centroid of the water areas in the masked image
stats = regionprops(BW, 'Centroid');
centroids = cat(1, stats.Centroid);
fprintf('Centroid of water areas: (%.2f, %.2f)\n', centroids(:,1), centroids(:,2));

% Calculate the total surface texture using gray level co-occurrence matrix (GLCM)
glcm = graycomatrix(gray_img, 'Offset', [0 1]);
statsGLCM = graycoprops(glcm);
fprintf('Contrast: %.2f\n', statsGLCM.Contrast);
fprintf('Correlation: %.2f\n', statsGLCM.Correlation);
fprintf('Energy: %.2f\n', statsGLCM.Energy);
fprintf('Homogeneity: %.2f\n', statsGLCM.Homogeneity);

% Calculate the water surface texture
glcm_Water = graycomatrix(Ieq);
statsGLCM = graycoprops(glcm_Water);
fprintf('Contrast of water: %.2f\n', statsGLCM.Contrast);
fprintf('Contrast of water: %.2f\n', statsGLCM.Correlation);
fprintf('Energy of water: %.2f\n', statsGLCM.Energy);
fprintf('Homogeneity of water: %.2f\n', statsGLCM.Homogeneity);

% Create a complement binary mask
land_mask = Ieq.*double(~BW);
imshow(land_mask);
title('land_mask');
figure;

% Calculate the land surface texture
glcmW_Land = graycomatrix(land_mask, 'Offset', [0 1]);
statsGLCM_Land = graycoprops(glcmW_Land);
fprintf('Contrast of land: %.2f\n', statsGLCM_Land.Contrast);
fprintf('Correlation of land: %.2f\n', statsGLCM_Land.Correlation);
fprintf('Energy of land: %.2f\n', statsGLCM_Land.Energy);
fprintf('Homogeneity of land: %.2f\n', statsGLCM_Land.Homogeneity);

% Threshold of pixel value
pixel_threshold_total = graythresh(gray_img);
fprintf('Total pixel value threshold contrast: %.2f\n', pixel_threshold_total);
pixel_threshold_totalC = graythresh(Ieq);
fprintf('Total pixel value threshold after contrast: %.2f\n', pixel_threshold_totalC);
pixel_threshold_water = graythresh(maskedImage);
fprintf('Water pixel value Threshold: %.2f\n', pixel_threshold_water);
pixel_threshold_land = graythresh(land_mask);
fprintf('Land pixel value Threshold: %.2f\n', pixel_threshold_land);

% Find the smallest water pixel value in the masked image
% unneccesary
% the smallest is 0, the largest is 1. It proves that the pixel value range
% of water is not limited
smallest_value_water = min(maskedImage(maskedImage > 0));
fprintf('Smallest water pixel value: %.2f\n', smallest_value_water);
largest_value_water = max(maskedImage(maskedImage < 255 ));
fprintf('Largest water pixel value: %.2f\n', largest_value_water);

% Find the area boundary
% unescessary since it doesn't show a visible region
boundaries = bwboundaries(BW, 'noholes');
plot(boundaries{1});
title('Coordinates of boundary points');

% plot the boundary without hole -> more accurate
for k = 1:length(boundaries)
    plot(boundaries{k}(:,2), boundaries{k}(:,1));
    title('Boundaries plot');
end







