cd iemocap/
for i in *.tar.gz; do
  tar -xvf $i
done

python iemocap_to_csv.py --output ./iemocap.csv --root ./IEMOCAP_full_release

cd IEMOCAP_full_release/

for i in Session*; do
  mv $i/dialog/transcriptions $i/
  mv $i/dialog/EmoEvaluation $i/
  rm -rf $i/dialog/*
  mv $i/transcriptions $i/dialog
  mv $i/EmoEvaluation $i/dialog

  mv $i/sentences/wav $i/
  rm -rf $i/sentences/*
  mv $i/wav $i/sentences
done
