% get access to the newest image
% eyeOfHorusPi = raspi("127.0.1.1", "pi", "raspberry");
% folderAddress = system(eyeOfHorusPi, "home/tracytran/Desktop/Camera_Image");
% newestUpdate = system(eyeOfHorusPi, "ls -t folderAddress | head -n 1");
% newestUpdateAddress =  system(eyeOfHorusPi, ...
%    "home/tracytran/Desktop/Camera_Image" + newestUpdate);
% importedImage = getFile(eyeOfHorusPi, newestUpdate);

% get access to laptop's images
folderAddress = system("C:/Users/tranp/OneDrive/Desktop/ARTS lab ..." + ...
    "    /Camera Project/Images/Creek image 2_1403.bmp");
getFile = fullfile(pwd, 'Creek image 2_1403.bmp');


% import to matlab
%img = imwrite(importedImage, newestUpdateAddress);
%imshow(img);



