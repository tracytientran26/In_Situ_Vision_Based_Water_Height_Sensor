clc
close all
clear all

% get the image
folderAddress = "C:\Users\tranp\OneDrive\Desktop\ARTS lab\Camera Project\Images";
fileAddress = "Creek image 2_1403.bmp";
get_img = fullfile(folderAddress, fileAddress);
imread(get_img);

% set the variable
x = get_img;

% get USGS dataset
folder_dataset = "C:\Users\tranp\OneDrive\Desktop\ARTS lab\Camera Project\Design";
file_dataset = "USGS data.py";
get_dataset = fullfile(folder_dataset, file_dataset);
imread(get_dataset);

% pair the images and data in the same timestamp

% split into training data and validation

% training setting

% build a regression model

% calculate errors and build improvement

% export numerical value




