# This code is partially borrowed from ACTOR's repo
rm -rf dataset

echo "The datasets will be stored in the 'dataset' folder\n"

echo "Downloading three datasets\n"
gdown "https://drive.google.com/uc?id=1CRTKMkAeZigjBZstjhn8gBhJpbO0bMzY"
echo "Extracting three dataset\n"
tar xfzv dataset.tar.gz
echo "Cleaning\n"
rm dataset.tar.gz

echo "Downloading done!"