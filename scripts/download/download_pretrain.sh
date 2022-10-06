# This code is partially borrowed from ACTOR's repo
rm -rf ckpt
echo "The pretrained model will be stored in the 'ckpt/pretrained' folder\n"

echo "Downloading the pre-trained models \n"
gdown "https://drive.google.com/uc?id=1V8HGHpcZzZxuKGS_gEvE4-k-yRzCcdvs"
echo "Extracting...\n"
tar xfzv ckpt.tar.gz
echo "Cleaning\n"
rm ckpt.tar.gz

echo "Done!"
