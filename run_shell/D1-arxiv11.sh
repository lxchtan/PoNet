WORKDIR=`pwd`
DATADIR=${WORKDIR}/datasets/arxiv-11
TMPDIR=`mktemp -d`
git clone https://github.com/LiqunW/Long-document-dataset ${TMPDIR}
cd ${TMPDIR}
RARDIR=rar_dir
mkdir ${RARDIR}
mv cs.*.rar math.*.rar ${RARDIR}/

unrar_dir(){
  src_path=`readlink -f $1`
  dst_path=`readlink -f $2`
  rar_files=`find $src_path -name '*.rar'`
  IFS=$'\n'; array=$rar_files; unset IFS
  for rar_file in $array; do
      file_path=`echo $rar_file | sed -e "s;$src_path;$dst_path;"`
      ext_path=${file_path%/*}
      if [ ! -d $ext_path ]; then
          mkdir -p $ext_path
      fi
      unrar x $rar_file $ext_path
  done
}

mkdir data
unrar_dir "${RARDIR}" "data"

if [ ! -f dataloader.py_ori ]; then
  cp dataloader.py dataloader.py_ori
  echo "Dataloader('data', 32)" >> dataloader.py
fi
python dataloader.py

mv Dataset.txt Labels_file.txt data/
mv data ${DATADIR}

# if [ -d ${TMPDIR} ]; then
#   rm -r ${TMPDIR}
# fi