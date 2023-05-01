#conda activate uretinex-net
cd python_src
dir="../test_img/*"
dirs=`find $dir -maxdepth 0 -type f -name *.png`
for i in $dirs;
do
    python uretinex-net.py --img_path $i
done

cd ../
#conda deactivate
