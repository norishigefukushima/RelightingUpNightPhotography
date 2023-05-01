function lr3m()

clear
addpath('../thirdparty/LR3M-Method/code/lowRank/');

addpath('../thirdparty/LR3M-Method/code');

para.beta    = 0.015;
para.omega   = 2;
para.lambda  = 2.5;
para.sigma   = 10;
para.gamma   = 2.2;
para.epsilon = 1;


par.nSig     =   para.sigma;
par.win      =   7;
par.nblk     =   40;
par.c1       =   2.9*sqrt(2);
par.gama     =   0.65;
par.delta    =   0.23;
par.Itr      =   2;
par.S        =   25;
par.step     =   min(6, par.win-1);
para.par     =   par;

files = dir('../test_img/*.png');

for i = 1:length(files)
    img_ = im2double(imread(['../test_img/' files(i).name]));
    img_ = im2gray(img_);
    img = img_;
    img(:,:,2) = img_;
    img(:,:,3) = img_;
    [I, R, L] = LR3M(img, para);
%     imwrite(I, ['../LR3M/I_gray_' files(i).name]);
    imwrite(R, ['../LR3M/R_gray_' files(i).name]);
%     imwrite(L, ['../LR3M/L_gray_' files(i).name]);
%     figure,imshow(I)
%     figure,imshow(R)
%     figure,imshow(L)
end