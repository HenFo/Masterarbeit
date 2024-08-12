cd meld/
for i in *.tar.gz; do
  tar -xvf $i
done
cd MELD.Raw/
for i in *.tar.gz; do
  tar -xvf $i
done


find . -type f -name '*_sent_emo.csv' -exec mv {} .. \; 
find . -maxdepth 1 -type f -name '*.*' -exec rm {} \;
find . -type f -name '*.*.mp4' -exec rm {} \;
find . -type f -name 'final_videos_test*.mp4' -exec rm {} \;

mkdir -p ../audio

mv *train*/ ../audio/train
mv *dev*/ ../audio/dev
mv *test*/ ../audio/test

cd ..
rm -r MELD.Raw
cd ..
sh convert_mp4_to_wav.sh